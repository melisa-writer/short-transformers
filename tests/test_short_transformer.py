# Runnable check for the analyse -> prune -> analyse loop on a tiny fake model.
#   python tests/test_short_transformer.py   (or pytest)
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from short_transformers import ShortTransformer
from short_transformers.utils import get_best_pruning_start, get_scored_blocks

HIDDEN, LAYERS = 4, 6


class Layer(nn.Module):
    def __init__(self, idx):
        super().__init__()
        self.self_attn = SimpleNamespace(layer_idx=idx)
        self.rot = nn.Linear(HIDDEN, HIDDEN, bias=False)

    def forward(self, hidden_states, **kw):
        return (hidden_states + self.rot(hidden_states),)


class Inner(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(10, HIDDEN)
        self.layers = nn.ModuleList(Layer(i) for i in range(LAYERS))


class FakeCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Inner()
        self.config = SimpleNamespace(_name_or_path="fake", num_hidden_layers=LAYERS)

    @property
    def device(self):
        return torch.device("cpu")

    def forward(self, input_ids, **kw):
        h = self.model.embed(input_ids)
        for layer in self.model.layers:
            h = layer(h)[0]
        return h


class Tok:
    def __call__(self, text, **kw):
        ids = torch.tensor([[ord(c) % 10 for c in text]])
        return SimpleNamespace(to=lambda dev: {"input_ids": ids})


def hidden_states(model, ids):
    """x[l] = input to layer l, x[L] = output of the last layer."""
    h = model.model.embed(ids)
    xs = [h]
    for layer in model.model.layers:
        h = layer(h)[0]
        xs.append(h)
    return xs


def test_result_rows_are_block_sizes():
    torch.manual_seed(0)
    model = ShortTransformer.from_model(FakeCausalLM())
    text = "abcdef"
    ds = [{"text": text}]
    with torch.no_grad():
        xs = hidden_states(model, Tok()(text, return_tensors="pt").to("cpu")["input_ids"])
    result = model.analyse_layers(dataset=ds, tokenizer=Tok(), key="text", limit=1)

    assert result.shape == (LAYERS + 1, LAYERS)
    assert np.all(result[0] == 0)
    for n in range(1, LAYERS + 1):
        for l in range(0, LAYERS - n + 1):
            expected = model.distance(xs[l], xs[l + n])
            assert abs(result[n, l] - expected) < 1e-6, (n, l)
        # cells past the last valid start layer are never written
        assert np.all(result[n, LAYERS - n + 1 :] == 0)


def test_best_start_matches_block_size():
    result = np.zeros((LAYERS + 1, LAYERS))
    result[2, :] = 1.0
    result[2, 4] = 0.1  # 2-layer block at layers 4-5 is the last valid start
    result[3, :] = 1.0
    result[3, 3] = 0.2
    assert get_best_pruning_start(result, block_size=2) == 4
    assert get_best_pruning_start(result, block_size=3) == 3
    stats = get_scored_blocks(result, return_md=False)
    assert stats[2] == {"start_layer": 4, "score": 0.1}
    assert stats[3] == {"start_layer": 3, "score": 0.2}


def test_prune_twice_and_reanalyse():
    torch.manual_seed(0)
    model = ShortTransformer.from_model(FakeCausalLM())
    ds = [{"text": "abcdef"}]

    model.prune(start_layer=1, block_size=2)
    assert model.layer_count == LAYERS - 2
    assert model.config.num_hidden_layers == LAYERS - 2
    assert [l.self_attn.layer_idx for l in model.model.layers] == list(range(LAYERS - 2))

    # analyse after prune: fresh memory, wrapped forwards use new indices
    result = model.analyse_layers(dataset=ds, tokenizer=Tok(), key="text", limit=1)
    assert result.shape == (LAYERS - 1, LAYERS - 2)
    assert np.all(np.isfinite(result))
    assert result[1, 0] > 0

    # pruning layer 0 must still reset memory on the next run
    model.prune(start_layer=0, block_size=1)
    assert model.layer_count == LAYERS - 3
    result = model.analyse_layers(dataset=ds, tokenizer=Tok(), key="text", limit=1)
    assert result.shape == (LAYERS - 2, LAYERS - 3)
    assert np.all(np.isfinite(result)) and result[1, 0] > 0


if __name__ == "__main__":
    test_result_rows_are_block_sizes()
    test_best_start_matches_block_size()
    test_prune_twice_and_reanalyse()
    print("ok")
