import torch
import torch.nn as nn

from transformers import AutoModel, AutoTokenizer


def get_teacher_model(model_name: str, block_size: int, start_layer: int):
    model = AutoModel.from_pretrained(
        model_name, output_attentions=False
    )

    # remove normalization
    model.norm = nn.Identity(model.config.hidden_size)

    # cut layers
    new_layers = nn.ModuleList()

    cut_from_inclusive = start_layer + block_size

    for i in range(0, len(model.layers)):
        if i < cut_from_inclusive:
            layer = model.layers[i]
            new_layers.append(layer)

    model.layers = new_layers

    changed_num_hidden_layers = cut_from_inclusive - 1
    changed_model_name_or_path = (
        f"{model.config._name_or_path}-{changed_num_hidden_layers}L"
    )
    # freeze everything
    model.eval()
    return model


class LinearApproximation(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.layer = nn.utils.parametrizations.orthogonal(
            nn.Linear(hidden_size, hidden_size, bias=True)
        )

    def forward(self, *args, **kwargs):
        hidden_states = args[0]
        hidden_states = self.layer(hidden_states)
        # @TODO residual - should we add a residual?
        # hidden_states = args[0] + hidden_states
        outputs = (hidden_states,)
        return outputs


def get_student_model(
    model_name: str, block_size: int, start_layer: int
):
    # @TODO: support caching
    model = AutoModel.from_pretrained(
        model_name, output_attentions=False, use_cache=False
    )

    # remove normalization
    model.norm = nn.Identity(model.config.hidden_size)

    # cut layers
    new_layers = nn.ModuleList()

    cut_from_inclusive = start_layer + block_size

    for i in range(0, len(model.layers)):
        if i < cut_from_inclusive:
            layer = model.layers[i]
            if i < start_layer:
                # freeze layers
                layer.requires_grad = False
                new_layers.append(layer)
            else:
                layer = LinearApproximation(model.config.hidden_size)
                new_layers.append(layer)
                # add just one linear layer
                # composition of linear layers is linear again
                break 

    model.layers = new_layers

    changed_num_hidden_layers = cut_from_inclusive - 1
    changed_model_name_or_path = (
        f"{model.config._name_or_path}-{changed_num_hidden_layers}L"
    )
    # freeze everything
    model.eval()
    return model

if __name__ == "__main__":

    model_name = "mistralai/Mistral-7B-v0.1"
    block_size = 1
    start_layer = 5

    student_model = get_student_model(model_name, block_size=block_size, start_layer=start_layer).to("cuda")
    teacher_model = get_teacher_model(model_name, block_size=block_size, start_layer=start_layer).to("cuda")

    print("STUDENT_MODEL")
    print(student_model)

    print("TEACHER_MODEL")
    print(teacher_model)

    try prediction
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    sample = tokenizer(
        "This is a sample string.",
        return_tensors="pt",
        padding=False,
        truncation=False,
    ).to("cuda")

    out = student_model(**sample)

"""
Outputs:

STUDENT_MODEL
Phi3Model(
  (embed_tokens): Embedding(32064, 3072, padding_idx=32000)
  (embed_dropout): Dropout(p=0.0, inplace=False)
  (layers): ModuleList(
    (0-4): 5 x Phi3DecoderLayer(
      (self_attn): Phi3Attention(
        (o_proj): Linear(in_features=3072, out_features=3072, bias=False)
        (qkv_proj): Linear(in_features=3072, out_features=9216, bias=False)
        (rotary_emb): Phi3RotaryEmbedding()
      )
      (mlp): Phi3MLP(
        (gate_up_proj): Linear(in_features=3072, out_features=16384, bias=False)
        (down_proj): Linear(in_features=8192, out_features=3072, bias=False)
        (activation_fn): SiLU()
      )
      (input_layernorm): Phi3RMSNorm()
      (resid_attn_dropout): Dropout(p=0.0, inplace=False)
      (resid_mlp_dropout): Dropout(p=0.0, inplace=False)
      (post_attention_layernorm): Phi3RMSNorm()
    )
    (5): LinearApproximation(
      (layer): ParametrizedLinear(
        in_features=3072, out_features=3072, bias=True
        (parametrizations): ModuleDict(
          (weight): ParametrizationList(
            (0): _Orthogonal()
          )
        )
      )
    )
  )
  (norm): Identity()
)
TEACHER_MODEL
Phi3Model(
  (embed_tokens): Embedding(32064, 3072, padding_idx=32000)
  (embed_dropout): Dropout(p=0.0, inplace=False)
  (layers): ModuleList(
    (0-5): 6 x Phi3DecoderLayer(
      (self_attn): Phi3Attention(
        (o_proj): Linear(in_features=3072, out_features=3072, bias=False)
        (qkv_proj): Linear(in_features=3072, out_features=9216, bias=False)
        (rotary_emb): Phi3RotaryEmbedding()
      )
      (mlp): Phi3MLP(
        (gate_up_proj): Linear(in_features=3072, out_features=16384, bias=False)
        (down_proj): Linear(in_features=8192, out_features=3072, bias=False)
        (activation_fn): SiLU()
      )
      (input_layernorm): Phi3RMSNorm()
      (resid_attn_dropout): Dropout(p=0.0, inplace=False)
      (resid_mlp_dropout): Dropout(p=0.0, inplace=False)
      (post_attention_layernorm): Phi3RMSNorm()
    )
  )
  (norm): Identity()
)
"""