from functools import partial, wraps

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from short_transformers.dist import angular_distance_last_token
from short_transformers.utils import get_best_pruning_start, get_logger

logger = get_logger("short-transformers", debug=True)


class Memory:
    def __init__(self, layer_count: int):
        self.examples_count: int = -1
        # result[n, l]: averaged distance for removing the n-layers block starting at layer l
        self.result = np.zeros((layer_count + 1, layer_count))
        self.layers_outputs: dict = {}


class ShortTransformer(PreTrainedModel):
    @classmethod
    def from_model(cls, model, distance=angular_distance_last_token):
        cls = model
        cls.layer_count = len(cls.model.layers)
        cls.distance = distance

        # add memory for storing intermediate layers outputs
        cls.memory = Memory(cls.layer_count)

        # @TODO add distances here
        # @TODO auto assign all methods from the class here
        cls.clear_memory = partial(ShortTransformer.clear_memory, cls)
        cls.analyse_layers = partial(ShortTransformer.analyse_layers, cls)
        cls.prune = partial(ShortTransformer.prune, cls)
        cls.remove_layers = partial(ShortTransformer.remove_layers, cls)
        cls.set_metric = partial(ShortTransformer.set_metric, cls)

        ShortTransformer._wrap_layers(cls)

        return cls

    @classmethod
    def from_pretrained(cls, *args, **kw):
        # @TODO: support other AutoModels variants
        model = AutoModelForCausalLM.from_pretrained(*args, **kw)
        return cls.from_model(model)

    @staticmethod
    def clear_memory(model) -> None:
        model.memory = Memory(model.layer_count)

    @staticmethod
    def _wrap_layers(model) -> None:
        # (re)wrap each layer forward with its current index; idempotent via __wrapped__
        for layer_idx, layer in enumerate(model.model.layers):
            forward = getattr(layer.forward, "__wrapped__", layer.forward)
            layer.forward = ShortTransformer._layer_io(model, layer_idx)(forward)

    @staticmethod
    def _layer_io(model, layer_idx: int):
        def decorator(f):
            @wraps(f)
            def wrap(*args, **kw):
                nonlocal model
                nonlocal layer_idx

                input_hidden_states = args[0] if args else kw["hidden_states"]

                if layer_idx == 0:
                    # clear the memory of previous example outputs and remmeber the input
                    model.memory.layers_outputs = {
                        -1: torch.clone(input_hidden_states).to("cpu")
                    }
                    model.memory.examples_count += 1

                # pass all arguments to the function
                result = f(*args, **kw)

                # decoder layers return a tuple in transformers < 4.54, a bare tensor since
                output_hidden_states = result[0] if isinstance(result, tuple) else result
                output_hidden_states = output_hidden_states.to("cpu")

                # calculate scores from -1 to this layer:
                for k, v in model.memory.layers_outputs.items():
                    dist = model.distance(v, output_hidden_states)

                    cut_layers = layer_idx - k

                    model.memory.result[cut_layers, k + 1] = (
                        model.memory.result[cut_layers, k + 1]
                        * model.memory.examples_count
                        + dist
                    ) / (model.memory.examples_count + 1)

                # remember the state
                model.memory.layers_outputs[layer_idx] = output_hidden_states
                return result

            return wrap

        return decorator

    @staticmethod
    def set_metric(model, criterion_callable):
        model.distance = criterion_callable

    @staticmethod
    def analyse_layers(
        model,
        dataset,
        tokenizer=None,
        use_chat_template=False,
        key: str = "content",
        limit: int = 1,
        max_length: int = 1000,
        batch_size: int = 1
    ) -> np.ndarray:
        # ponytail: metrics take one sequence at a time; batching needs pad-aware metrics
        assert batch_size == 1, "batch_size > 1 is not supported yet."
        if tokenizer is None:
            logger.debug(
                "Tokenizer not provided, will load tokenizer from config._name_or_path"
            )
            try:
                tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
            except Exception as e:
                raise RuntimeError(
                    f"Loading the tokenizer failed with error: {e}.\nUse analyse_layers(... tokenizer=...) to manually set the tokenizer."
                ) from e

        logger.debug(f"Running inference on {limit} samples.")

        model.model.eval()

        with torch.no_grad():
            count = 0
            for d in tqdm(dataset):
                content = d[key]
                if use_chat_template:
                    inputs = tokenizer.apply_chat_template(
                        content,
                        tokenize=True,
                        add_generation_prompt=False,
                        return_tensors="pt",
                        return_dict=True,
                        truncation=True,
                        max_length=max_length,
                    ).to(model.device)
                else:
                    inputs = tokenizer(
                        content,
                        return_tensors="pt",
                        truncation=True,
                        max_length=max_length,
                    ).to(model.device)
                model(**inputs)
                count += 1
                if count >= limit:
                    break
        result = model.memory.result
        model.clear_memory()
        return result

    @staticmethod
    def prune(model, start_layer: int, block_size: int):
        assert (
            0 <= start_layer and start_layer + block_size <= model.layer_count
        ), f"Block {start_layer}-{start_layer + block_size - 1} is out of range for {model.layer_count} layers."

        removed_layers = range(start_layer, start_layer + block_size)
        logger.debug(f"Removing layers: {list(removed_layers)}")

        new_layers = torch.nn.ModuleList()
        for i, layer in enumerate(model.model.layers):
            if i in removed_layers:
                continue
            layer.self_attn.layer_idx = len(new_layers)
            new_layers.append(layer)

        model.model.layers = new_layers
        model.layer_count = len(new_layers)
        model.clear_memory()
        ShortTransformer._wrap_layers(model)

        changed_num_hidden_layers = model.layer_count
        changed_model_name_or_path = (
            f"{model.config._name_or_path}-{changed_num_hidden_layers}L"
        )

        logger.debug(f"""Changing model config to reflect changes:
        config.num_hidden_layers: {model.config.num_hidden_layers} -> {changed_num_hidden_layers}
        config._name_or_path: {model.config._name_or_path} -> {changed_model_name_or_path}""")

        model.config.num_hidden_layers = changed_num_hidden_layers
        model.config._name_or_path = changed_model_name_or_path

        return model

    @staticmethod
    def remove_layers(
        model,
        block_size,
        dataset,
        tokenizer=None,
        use_chat_template=False,
        key="text",
        limit=1,
        batch_size=1,
        max_length=1000,
    ):
        result = model.analyse_layers(
            dataset=dataset,
            tokenizer=tokenizer,
            use_chat_template=use_chat_template,
            key=key,
            limit=limit,
            max_length=max_length,
            batch_size=batch_size,
        )
        logger.debug(f"Choosing optimal {block_size}-layers block to prune.")
        start_layer = get_best_pruning_start(result=result, block_size=block_size)
        logger.debug(f"Best {block_size}-layers block to prune starts at layer: {start_layer}.")
        return model.prune(start_layer=start_layer, block_size=block_size)
