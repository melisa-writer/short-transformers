import copy
from functools import partial, wraps

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, PreTrainedModel

from short_transformers.dist import (
    angular_distance_all_tokens,
    angular_distance_last_token,
)


class ShortTransformer(PreTrainedModel):

    @classmethod
    def from_model(cls, model):
        cls = model

        # add memory for storing intermediate layers outputs
        cls.layer_count = len(cls.model.layers)
        cls.memory = {
            "examples_count": -1,
            "result": np.zeros((cls.layer_count, cls.layer_count)),
            "layers_outputs": {},
        }
        # @TODO add distances here
        # @TODO auto assign all methods from the class here
        cls.clear_memory = partial(ShortTransformer.clear_memory, cls)
        cls.save_memory = partial(ShortTransformer.save_memory, cls.memory)
        cls.analyse_layers = partial(ShortTransformer.analyse_layers, cls)
        cls.get_optimal_cut = partial(ShortTransformer.get_optimal_cut, cls)
        cls.cut = partial(ShortTransformer.cut, cls)
        cls.remove_layers = partial(ShortTransformer.remove_layers, cls)

        # add decorators to each forward in layers
        for layer_idx, layer in enumerate(cls.model.layers):
            layer.forward = ShortTransformer._layer_io(cls.memory, layer_idx)(
                layer.forward
            )

        return cls

    @classmethod
    def from_pretrained(cls, *args, **kw):
        # @TODO: support other AutoModels variants
        model = AutoModelForCausalLM.from_pretrained(*args, **kw)
        return cls.from_model(model)

        # # add memory for storing intermediate layers outputs
        # cls.layer_count = len(cls.model.layers)
        # cls.memory = {
        #     "examples_count": -1,
        #     "result": np.zeros((cls.layer_count, cls.layer_count)),
        #     "layers_outputs": {},
        # }
        # # @TODO add distances here
        # cls.clear_memory = partial(ShortTransformer.clear_memory, cls)
        # cls.save_memory = partial(ShortTransformer.save_memory, cls.memory)
        # cls.analyse_layers = partial(ShortTransformer.analyse_layers, cls)
        # cls.get_optimal_cut = partial(ShortTransformer.get_optimal_cut, cls)
        # cls.cut = partial(ShortTransformer.cut, cls)

        # # add decorators to each forward in layers
        # for layer_idx, layer in enumerate(cls.model.layers):
        #     layer.forward = ShortTransformer._layer_io(cls.memory, layer_idx)(layer.forward)

        # return cls

    @staticmethod
    def clear_memory(model):
        model.memory = {
            "examples_count": -1,
            "result": np.zeros((cls.layer_count, cls.layer_count)),
            "layers_outputs": {},
        }

    @staticmethod
    def save_memory(memory, file_path):
        np.savez(file_path, **memory)

    @staticmethod
    def _layer_io(memory, layer_idx):
        def decorator(f):
            @wraps(f)
            def wrap(*args, **kw):
                nonlocal memory
                nonlocal layer_idx

                input_hidden_states = args[0]

                if layer_idx == 0:
                    # clear the memory of previous example outputs and remmeber the input
                    memory["layers_outputs"] = {
                        -1: torch.clone(input_hidden_states).to("cpu")
                    }
                    memory["examples_count"] += 1

                # pass all arguments to the function
                result = f(*args, **kw)

                # calculate io metric for all layers:
                output_hidden_states = torch.clone(result[0]).to("cpu")

                # calculate scores from -1 to this layer:
                for k, v in memory["layers_outputs"].items():
                    dist = angular_distance_last_token(v, output_hidden_states)

                    cut_layers = layer_idx - k - 1

                    memory["result"][cut_layers, k + 1] = (
                        memory["result"][cut_layers, k + 1] * memory["examples_count"]
                        + dist
                    ) / (memory["examples_count"] + 1)

                # remember the state
                memory["layers_outputs"][layer_idx] = torch.clone(
                    output_hidden_states
                ).to("cpu")
                return result

            return wrap

        return decorator

    @staticmethod
    def analyse_layers(
        model, tokenizer, dataset, key="content", limit=1, max_length=1000
    ):
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

    @staticmethod
    def get_optimal_cut(model, num):
        assert (
            num < model.layer_count and num > 0
        ), f"Expected `num` value between 1 and {model.layer_count -1}, got {num}."
        layer_result = model.memory["result"][num, : model.layer_count - num]
        return np.argmin(layer_result)

    @staticmethod
    def cut(model, start_layer, n):
        new_layers = torch.nn.ModuleList()

        remove_layers = list(range(start_layer, start_layer + n))
        print(f"Removing layers: {remove_layers}")

        count = 0
        for i in range(0, model.num_of_layers):
            if i not in remove_layers:
                count += 1
                layer = model.model.layers[i]
                layer.layer_idx = count
                layer.self_attn.layer_idx = count
                new_layers.append(layer)
            else:
                print(f"skipping layer: {i}")

        copyOfModel = copy.deepcopy(model)
        copyOfModel.model.layers = new_layers

        copyOfModel.config.num_hidden_layers = model.num_of_layers - n
        # copyOfModel.save_pretrained("short_mistral")
        # model.model = copyOfModel
        return ShortTransformer.from_model(copyOfModel)

    @staticmethod
    def remove_layers(
        model,
        tokenizer,
        n,
        dataset,
        key="text",
        limit=1,
        batch_size=1,
        max_length=1000,
        return_outputs=False,
        distance=angular_distance_last_token,
    ):

        model.analyse_layers(tokenizer, dataset, key, limit, max_length)
        start_layer = model.get_optimal_cut(num=n)
        return model.cut(start_layer, n)
