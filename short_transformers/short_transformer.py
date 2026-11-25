from functools import partial, wraps

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from short_transformers.dist import angular_distance_last_token
from short_transformers.utils import get_best_pruning_start, get_logger

# raise the level from the application: logging.getLogger("short-transformers").setLevel(logging.DEBUG)
logger = get_logger("short-transformers")

# where the decoder layers live: causal LMs, then multimodal models wrapping a language model
LAYER_PATHS = (
    "model.layers",
    "model.language_model.layers",
    "model.language_model.model.layers",
    "language_model.model.layers",
    "language_model.layers",
)


def _layers_owner(model):
    """Return the module whose `layers` attribute is the decoder ModuleList."""
    for path in LAYER_PATHS:
        owner = model
        for attr in path.split(".")[:-1]:
            owner = getattr(owner, attr, None)
            if owner is None:
                break
        if isinstance(getattr(owner, "layers", None), torch.nn.ModuleList):
            return owner
    raise ValueError(f"Could not find the decoder layers, tried: {', '.join(LAYER_PATHS)}.")


def _text_config(config):
    return config.get_text_config() if hasattr(config, "get_text_config") else config


def _layer_schedule(model, layer_count: int) -> list:
    """The order the layers run in one forward pass, as (layer index, pass) pairs.

    One entry per layer for ordinary models. Looped models (e.g. Inkling 2: layers
    loop_layer_index_start..loop_layer_index_end-1 run loop_num_passes times) get one entry
    per layer application, so each pass through a looped layer is its own position.
    """
    cfg = _text_config(model.config)
    start = getattr(cfg, "loop_layer_index_start", None)
    end = getattr(cfg, "loop_layer_index_end", None)
    passes = getattr(cfg, "loop_num_passes", 1) or 1
    if start is None or end is None or passes <= 1 or not 0 <= start < end <= layer_count:
        return [(i, 0) for i in range(layer_count)]
    return (
        [(i, 0) for i in range(start)]
        + [(i, p) for p in range(passes) for i in range(start, end)]
        + [(i, 0) for i in range(end, layer_count)]
    )


def _schedule_labels(schedule: list) -> list:
    looped = {i for i, p in schedule if p > 0}
    return [f"{i}p{p + 1}" if i in looped else str(i) for i, p in schedule]


class Memory:
    def __init__(self, position_count: int):
        self.examples_count: int = -1
        # result[n, l]: averaged distance for removing the n-layers block starting at position l
        # (positions are layers, or layer applications for looped models: see _layer_schedule)
        self.result = np.zeros((position_count + 1, position_count))
        # inputs[p]: the hidden states that entered position p in the current example
        self.inputs: list = []


class ShortTransformer(PreTrainedModel):
    @classmethod
    def from_model(cls, model, distance=angular_distance_last_token):
        cls = model
        cls.layer_count = len(_layers_owner(cls).layers)
        cls.distance = distance
        ShortTransformer._set_schedule(cls)

        # add memory for storing intermediate layers outputs
        cls.memory = Memory(len(cls.layer_schedule))

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
        try:
            model = AutoModelForCausalLM.from_pretrained(*args, **kw)
        except ValueError as e:
            # multimodal models (e.g. Inkling 2) only register an image-text-to-text class
            from transformers import AutoModelForImageTextToText

            logger.debug(f"AutoModelForCausalLM failed ({e}), trying AutoModelForImageTextToText.")
            model = AutoModelForImageTextToText.from_pretrained(*args, **kw)
        return cls.from_model(model)

    @staticmethod
    def _set_schedule(model) -> None:
        model.layer_schedule = _layer_schedule(model, model.layer_count)
        model.layer_labels = _schedule_labels(model.layer_schedule)
        model.is_looped = len(model.layer_schedule) != model.layer_count

    @staticmethod
    def clear_memory(model) -> None:
        model.memory = Memory(len(model.layer_schedule))

    @staticmethod
    def _wrap_layers(model) -> None:
        # (re)wrap each layer forward with its current index; idempotent via _short_original
        # (not __wrapped__: transformers' own forward decorators set it to the unbound function)
        for layer_idx, layer in enumerate(_layers_owner(model).layers):
            forward = getattr(layer.forward, "_short_original", layer.forward)
            layer.forward = ShortTransformer._layer_io(model, layer_idx)(forward)

    @staticmethod
    def _layer_io(model, layer_idx: int):
        def decorator(f):
            @wraps(f)
            def wrap(*args, **kw):
                nonlocal model
                nonlocal layer_idx

                input_hidden_states = args[0] if args else kw["hidden_states"]
                memory = model.memory

                if layer_idx == 0:
                    # a new forward pass: forget the previous example's states
                    memory.inputs = []
                    memory.examples_count += 1

                # position of this call in the forward pass; differs from layer_idx for looped models
                position = len(memory.inputs)
                expected = model.layer_schedule[position][0] if position < len(model.layer_schedule) else None
                if expected != layer_idx:
                    raise RuntimeError(
                        f"Layer {layer_idx} ran at position {position}, expected layer {expected}: the model "
                        "runs its layers in an order short-transformers does not know about."
                    )

                # remember what entered this position: a block's input is not always the previous
                # output (looped models restart each pass from the pre-loop state)
                memory.inputs.append(input_hidden_states.detach().to("cpu", copy=True))

                # pass all arguments to the function
                result = f(*args, **kw)

                # decoder layers return a tuple in transformers < 4.54, a bare tensor since
                output_hidden_states = result[0] if isinstance(result, tuple) else result
                output_hidden_states = output_hidden_states.detach().to("cpu")

                # score every block that ends here: positions start..position
                for start, block_input in enumerate(memory.inputs):
                    dist = model.distance(block_input, output_hidden_states)
                    block_size = position - start + 1

                    memory.result[block_size, start] = (
                        memory.result[block_size, start] * memory.examples_count + dist
                    ) / (memory.examples_count + 1)

                return result

            wrap._short_original = f
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

        model.eval()

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
    def _check_prunable(model) -> None:
        if model.is_looped:
            raise NotImplementedError(
                "Pruning looped models is not supported: their layers run several times per forward pass "
                "and analyse_layers() scores layer applications (see model.layer_labels), not layers."
            )

    @staticmethod
    def prune(model, start_layer: int, block_size: int):
        ShortTransformer._check_prunable(model)
        assert (
            0 <= start_layer and start_layer + block_size <= model.layer_count
        ), f"Block {start_layer}-{start_layer + block_size - 1} is out of range for {model.layer_count} layers."

        removed_layers = range(start_layer, start_layer + block_size)
        logger.debug(f"Removing layers: {list(removed_layers)}")

        owner = _layers_owner(model)
        new_layers = torch.nn.ModuleList()
        for i, layer in enumerate(owner.layers):
            if i in removed_layers:
                continue
            layer.self_attn.layer_idx = len(new_layers)
            new_layers.append(layer)

        owner.layers = new_layers
        model.layer_count = len(new_layers)
        ShortTransformer._set_schedule(model)
        model.clear_memory()
        ShortTransformer._wrap_layers(model)

        changed_num_hidden_layers = model.layer_count
        changed_model_name_or_path = (
            f"{model.config._name_or_path}-{changed_num_hidden_layers}L"
        )

        logger.debug(f"""Changing model config to reflect changes:
        config.num_hidden_layers: {_text_config(model.config).num_hidden_layers} -> {changed_num_hidden_layers}
        config._name_or_path: {model.config._name_or_path} -> {changed_model_name_or_path}""")

        _text_config(model.config).num_hidden_layers = changed_num_hidden_layers
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
        ShortTransformer._check_prunable(model)  # before the (long) analysis
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
