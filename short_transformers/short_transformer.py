from functools import partial, wraps

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from short_transformers.dist import (
    angular_distance_all_tokens,
    angular_distance_last_token,
)
from short_transformers.utils import (
    get_best_pruning_start,
    get_logger
)

logger = get_logger("short-transformers", debug=True)


class Memory:
    def __init__(self, layer_count: int):
        self.examples_count: int = -1
        self.result = np.zeros((layer_count, layer_count))
        self.layers_outputs: dict = {}


class ShortTransformer(PreTrainedModel):
    default_io_distance = angular_distance_last_token

    @classmethod
    def from_model(cls, model):
        cls = model
        cls.layer_count = len(cls.model.layers)

        # add memory for storing intermediate layers outputs
        cls.memory = Memory(cls.layer_count)

        # @TODO add distances here
        # @TODO auto assign all methods from the class here
        cls.clear_memory = partial(ShortTransformer.clear_memory, cls)
        cls.analyse_layers = partial(ShortTransformer.analyse_layers, cls)
        cls.prune = partial(ShortTransformer.prune, cls)
        cls.remove_layers = partial(ShortTransformer.remove_layers, cls)

        # add decorators to each forward in layers
        for layer_idx, layer in enumerate(cls.model.layers):
            layer.forward = ShortTransformer._layer_io(
                cls.memory, layer_idx, ShortTransformer.default_io_distance
            )(layer.forward)

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
    def _layer_io(memory, layer_idx: int, io_distance):
        def decorator(f):
            @wraps(f)
            def wrap(*args, **kw):
                nonlocal memory
                nonlocal layer_idx
                nonlocal io_distance

                input_hidden_states = args[0]

                if layer_idx == 0:
                    # clear the memory of previous example outputs and remmeber the input
                    memory.layers_outputs = {
                        -1: torch.clone(input_hidden_states).to("cpu")
                    }
                    memory.examples_count += 1

                # pass all arguments to the function
                result = f(*args, **kw)

                # calculate io metric for all layers:
                output_hidden_states = torch.clone(result[0]).to("cpu")

                # calculate scores from -1 to this layer:
                for k, v in memory.layers_outputs.items():
                    dist = io_distance(v, output_hidden_states)

                    cut_layers = layer_idx - k - 1

                    memory.result[cut_layers, k + 1] = (
                        memory.result[cut_layers, k + 1] * memory.examples_count + dist
                    ) / (memory.examples_count + 1)

                # remember the state
                memory.layers_outputs[layer_idx] = torch.clone(output_hidden_states).to(
                    "cpu"
                )
                return result

            return wrap

        return decorator

    # @TODO
    # @staticmethod
    # def set_pruning_criterion(model, criterion_callable):

    @staticmethod
    def analyse_layers(
        model,
        dataset,
        tokenizer=None,
        key: str = "content",
        limit: int = 1,
        max_length: int = 1000,
    ) -> None:
        if tokenizer is None:
            logger.debug(
                "Tokenizer not provided, will load tokenizer from config._name_or_path"
            )
            try:
                tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
            except Exception as e:
                logger.error(
                    f"Loading the tokenizer failed wwth error: {e}.\nUse analyse_layers(... tokenizer=...) to manually set the tokenizer."
                )
                raise RuntimeError

        logger.debug(f"Running inference on {limit} samples.")

        model.model.eval()
        with torch.no_grad():
            count = 0
            for d in tqdm(dataset):
                content = d[key]
                inputs = tokenizer(
                    content,
                    return_tensors="pt",
                    padding=False,
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
        new_layers = torch.nn.ModuleList()

        remove_layers = list(range(start_layer, start_layer + block_size))
        logger.debug(f"Removing layers: {remove_layers}")

        count = 0
        for i in range(0, model.layer_count):
            if i not in remove_layers:
                count += 1
                layer = model.model.layers[i]
                layer.layer_idx = count
                layer.self_attn.layer_idx = count
                new_layers.append(layer)

        model.model.layers = new_layers

        changed_num_hidden_layers = model.layer_count - block_size
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
        tokenizer,
        block_size,
        dataset,
        key="text",
        limit=1,
        batch_size=1,
        max_length=1000,
        return_outputs=False,
        distance=angular_distance_last_token,
    ):
        assert batch_size == 1, "batch_size > 1 is not supported yet."
        result = model.analyse_layers(
            dataset=dataset,
            tokenizer=tokenizer,
            key=key,
            limit=limit,
            max_length=max_length,
        )
        logger.debug(f"Choosing optimal {block_size}-layers block to prune.")
        start_layer = get_best_pruning_start(result=result, block_size=block_size)
        logger.debug(f"Best 5-layers block to prune starts at layer: {start_layer}.")
        return model.prune(start_layer=start_layer, block_size=block_size)
