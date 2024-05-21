import torch
import torch.nn as nn

from transformers import AutoModel, AutoTokenizer
import copy

def merge_layers(layerA, layerB):

    sdA = layerA.state_dict()
    sdB = layerB.state_dict()

    # Average all parameters
    for key in sdA:
        print(key)
        sdB[key] = (sdB[key] + sdA[key]) / 2.

    # Load averaged state_dict
    layerC = copy.deepcopy(layerA)
    layerC.load_state_dict(sdB)

    return layerC


def get_caterpillar_model(model_name: str, cut_layer: int, duplicate_layer: int):

    assert cut_layer!=duplicate_layer
    assert cut_layer!=0

    model = AutoModel.from_pretrained(
        model_name, output_attentions=False
    )

    # cut one layer, duplicate another
    new_layers = nn.ModuleList()

    for i in range(0, len(model.layers)):
        if i == cut_layer:
            # merge with previous
            prev_layer = new_layers[-1]
            avg_layer = merge_layers(model.layers[i], prev_layer)
            # overwrite prev layer
            new_layers[-1] = avg_layer

        elif i == duplicate_layer:
            # add twice - once average, once original
            layer = model.layers[i]
            layer_cloned = copy.deepcopy(layer)

            prev_layer = new_layers[-1]
            avg_layer = merge_layers(model.layers[i], prev_layer)
            
            new_layers.append(avg_layer)
            new_layers.append(layer_cloned)            
        else:
            layer = model.layers[i]
            new_layers.append(layer)

    model.layers = new_layers

    # rename layers
    # not sure how universal is this
    for i, layer in enumerate(model.layers):
        layer.self_attn.layer_idx = i

    changed_model_name_or_path = (
        f"{model.config._name_or_path}-updown_simple"
    )
    return model

if __name__ == "__main__":

    model_name = "mistralai/Mistral-7B-v0.1"
    cut_layer = 3
    duplicate_layer = 26

    model = get_caterpillar_model(model_name, cut_layer=cut_layer, duplicate_layer=duplicate_layer).to("cuda")

    print("CATERPILLAR_MODEL")
    print(model)

    # try prediction
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    sample = tokenizer(
        "This is a sample string.",
        return_tensors="pt",
        padding=False,
        truncation=False,
    ).to("cuda")

    out = model(**sample)