# Runnable check for the analyse -> prune -> analyse loop on a tiny fake model.
#   python tests/test_short_transformer.py   (or pytest)
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from short_transformers import ShortTransformer
from short_transformers.dist import relative_magnitude
from short_transformers.utils import get_best_pruning_start, get_scored_blocks

HIDDEN, LAYERS = 4, 6


class Layer(nn.Module):
    def __init__(self, idx):
        super().__init__()
        self.self_attn = SimpleNamespace(layer_idx=idx)
        self.rot = nn.Linear(HIDDEN, HIDDEN, bias=False)

    def forward(self, hidden_states, **kw):
        return (hidden_states + self.rot(hidden_states),)


class TensorLayer(Layer):
    # transformers >= 4.54 decoder layers return the tensor, not a tuple
    def forward(self, hidden_states, **kw):
        return super().forward(hidden_states)[0]


class Inner(nn.Module):
    def __init__(self, layer_cls=Layer):
        super().__init__()
        self.embed = nn.Embedding(10, HIDDEN)
        self.layers = nn.ModuleList(layer_cls(i) for i in range(LAYERS))


class FakeCausalLM(nn.Module):
    def __init__(self, layer_cls=Layer):
        super().__init__()
        self.model = Inner(layer_cls)
        self.config = SimpleNamespace(_name_or_path="fake", num_hidden_layers=LAYERS)

    @property
    def device(self):
        return torch.device("cpu")

    def forward(self, input_ids, **kw):
        h = self.model.embed(input_ids)
        for layer in self.model.layers:
            h = layer(h)
            h = h[0] if isinstance(h, tuple) else h
        return h


class Tok:
    def __call__(self, text, **kw):
        ids = torch.tensor([[ord(c) % 10 for c in text]])
        return SimpleNamespace(to=lambda dev: {"input_ids": ids})

    def apply_chat_template(self, messages, **kw):
        assert kw.get("return_dict") and kw.get("return_tensors") == "pt", kw
        return self("".join(m["content"] for m in messages))


def hidden_states(model, ids):
    """x[l] = input to layer l, x[L] = output of the last layer."""
    h = model.model.embed(ids)
    xs = [h]
    for layer in model.model.layers:
        h = layer(h)
        h = h[0] if isinstance(h, tuple) else h
        xs.append(h)
    return xs


def test_result_rows_are_block_sizes(layer_cls=Layer):
    torch.manual_seed(0)
    model = ShortTransformer.from_model(FakeCausalLM(layer_cls))
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


def test_layers_returning_bare_tensor():
    test_result_rows_are_block_sizes(TensorLayer)


def test_real_llama_roundtrip(tmp_path=None):
    import tempfile

    from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=4,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=32,
    )
    model = ShortTransformer.from_model(LlamaForCausalLM(cfg))
    ids = torch.arange(6)[None]
    with torch.no_grad():
        ref = model(input_ids=ids, output_hidden_states=True).hidden_states

    class IdsTok:
        def __call__(self, text, **kw):
            return SimpleNamespace(to=lambda dev: {"input_ids": ids})

    result = model.analyse_layers(dataset=[{"text": "x"}], tokenizer=IdsTok(), key="text", limit=1)
    assert result.shape == (5, 4)
    # hidden_states[l] is the input to layer l; the last entry is post-norm, so stop before it
    for n in range(1, 4):
        for l in range(0, 4 - n):
            expected = model.distance(ref[l], ref[l + n])
            assert abs(result[n, l] - expected) < 1e-5, (n, l)

    model.prune(start_layer=1, block_size=2)
    with torch.no_grad():
        model(input_ids=ids)
    out_dir = tmp_path or tempfile.mkdtemp()
    model.save_pretrained(out_dir)
    reloaded = AutoModelForCausalLM.from_pretrained(out_dir)
    assert reloaded.config.num_hidden_layers == 2
    assert len(reloaded.model.layers) == 2


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


def test_chat_template_path_matches_plain_path():
    torch.manual_seed(0)
    model = ShortTransformer.from_model(FakeCausalLM())
    plain = model.analyse_layers(dataset=[{"text": "abcdef"}], tokenizer=Tok(), key="text")
    chat = model.analyse_layers(
        dataset=[{"messages": [{"role": "user", "content": "abc"}, {"role": "assistant", "content": "def"}]}],
        tokenizer=Tok(), use_chat_template=True, key="messages",
    )
    assert np.allclose(plain, chat)


def test_chat_template_with_real_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {w: i for i, w in enumerate(["[UNK]", "<s>", "</s>", "hi", "there", "user", "assistant"])}
    raw = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]", bos_token="<s>", eos_token="</s>")
    tok.chat_template = "{% for m in messages %}{{ m['role'] }} {{ m['content'] }} {% endfor %}"

    torch.manual_seed(0)
    model = ShortTransformer.from_model(FakeCausalLM())
    result = model.analyse_layers(
        dataset=[{"messages": [{"role": "user", "content": "hi there"}]}],
        tokenizer=tok, use_chat_template=True, key="messages", max_length=3,
    )
    assert result.shape == (LAYERS + 1, LAYERS) and result[1, 0] > 0


def test_remove_layers_loads_tokenizer_from_config():
    from unittest.mock import patch

    torch.manual_seed(0)
    model = ShortTransformer.from_model(FakeCausalLM())
    with patch("short_transformers.short_transformer.AutoTokenizer") as auto:
        auto.from_pretrained.return_value = Tok()
        short = model.remove_layers(block_size=2, dataset=[{"text": "abcdef"}], key="text")
    auto.from_pretrained.assert_called_once_with("fake")
    assert short.layer_count == LAYERS - 2


def test_batch_size_above_one_is_rejected():
    model = ShortTransformer.from_model(FakeCausalLM())
    for call in (model.analyse_layers, lambda **kw: model.remove_layers(block_size=1, **kw)):
        try:
            call(dataset=[{"text": "abcdef"}], tokenizer=Tok(), key="text", batch_size=2)
        except AssertionError as e:
            assert "batch_size" in str(e)
        else:
            raise AssertionError("batch_size=2 was accepted")


def test_relative_magnitude_is_paper_ratio():
    x = torch.randn(1, 5, HIDDEN)
    identity = relative_magnitude(x, x)
    shrink = relative_magnitude(x, x - x / 2)  # f(x) = -x/2 -> ||f|| / ||x+f|| = 1
    assert abs(identity) < 1e-5
    assert abs(shrink - 1.0) < 1e-4
    assert identity < shrink


if __name__ == "__main__":
    test_batch_size_above_one_is_rejected()
    test_chat_template_path_matches_plain_path()
    test_chat_template_with_real_tokenizer()
    test_remove_layers_loads_tokenizer_from_config()
    test_relative_magnitude_is_paper_ratio()
    test_result_rows_are_block_sizes()
    test_layers_returning_bare_tensor()
    test_real_llama_roundtrip()
    test_best_start_matches_block_size()
    test_prune_twice_and_reanalyse()
    print("ok")
