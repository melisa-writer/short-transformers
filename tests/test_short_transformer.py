# Runnable check for the analyse -> prune -> analyse loop on a tiny fake model.
#   python tests/test_short_transformer.py   (or pytest)
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from short_transformers import ShortTransformer
from short_transformers.dist import (
    get_angular_distance_ith_token,
    get_euclidian_dist_ith_token,
    get_linear_approximation_ith_token,
    relative_magnitude,
)
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


def test_import_leaves_root_logger_alone():
    import logging
    import subprocess
    import sys

    code = (
        "import logging; logging.getLogger().addHandler(logging.NullHandler());"
        "import short_transformers; print(len(logging.getLogger().handlers))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "1"
    assert logging.getLogger("short-transformers").level == logging.WARN


def test_ith_token_metrics_on_bf16_and_short_sequences():
    x = torch.randn(1, 3, HIDDEN, dtype=torch.bfloat16)
    y = torch.randn(1, 3, HIDDEN, dtype=torch.bfloat16)
    for factory in (get_angular_distance_ith_token, get_linear_approximation_ith_token, get_euclidian_dist_ith_token):
        assert isinstance(factory(-1)(x, y), float)
        try:
            factory(5)(x, y)
        except RuntimeError as e:
            assert "6 tokens" in str(e) and isinstance(e.__cause__, IndexError)
        else:
            raise AssertionError("no error on too short sequence")
    # log of the mean squared residual, not of the sum
    xf, yf = x.float(), y.float()
    a = (xf[0, -1] @ yf[0, -1]) / (xf[0, -1] @ xf[0, -1])
    expected = float(torch.log(((a * xf[0, -1] - yf[0, -1]) ** 2).mean()))
    assert abs(get_linear_approximation_ith_token(-1)(x, y) - expected) < 1e-4


def test_plots_close_their_figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import tempfile

    from short_transformers.utils import draw_diagram, draw_layers_heatmap

    r = np.zeros((LAYERS + 1, LAYERS))
    for n in range(1, LAYERS + 1):
        r[n, : LAYERS - n + 1] = np.random.rand(LAYERS - n + 1) + n
    d = tempfile.mkdtemp()
    draw_diagram(r, f"{d}/d.png", title="t")
    draw_layers_heatmap(r, "m", "t", f"{d}/h.png")
    assert plt.get_fignums() == []


def test_relative_magnitude_is_paper_ratio():
    x = torch.randn(1, 5, HIDDEN)
    identity = relative_magnitude(x, x)
    shrink = relative_magnitude(x, x - x / 2)  # f(x) = -x/2 -> ||f|| / ||x+f|| = 1
    assert abs(identity) < 1e-5
    assert abs(shrink - 1.0) < 1e-4
    assert identity < shrink


# Looped models (Inkling 2): layers LOOP_START..LOOP_END-1 run PASSES times; every pass restarts
# from the pre-loop state, so a block's input is not always the previous layer's output.
LOOP_START, LOOP_END, PASSES = 2, 4, 3
SCHEDULE = [0, 1] + [2, 3] * PASSES + [4, 5]
LABELS = ["0", "1", "2p1", "3p1", "2p2", "3p2", "2p3", "3p3", "4", "5"]


class LoopedInner(nn.Module):
    def __init__(self, passes=PASSES):
        super().__init__()
        self.passes = passes
        self.embed = nn.Embedding(10, HIDDEN)
        # multimodal layout: the decoder sits under model.language_model
        self.language_model = nn.Module()
        self.language_model.layers = nn.ModuleList(TensorLayer(i) for i in range(LAYERS))
        self.message = nn.Linear(HIDDEN, HIDDEN, bias=False)

    def forward(self, ids, trace=None):
        layers = self.language_model.layers

        def run(layer, h):
            out = layer(h)
            if trace is not None:
                trace.append((h, out))
            return out

        h = self.embed(ids)
        for layer in layers[:LOOP_START]:
            h = run(layer, h)
        pre_loop = h
        for p in range(self.passes):
            # restart from the pre-loop state, mixed with a message from the previous pass
            h = pre_loop if p == 0 else pre_loop + self.message(h)
            for layer in layers[LOOP_START:LOOP_END]:
                h = run(layer, h)
        for layer in layers[LOOP_END:]:
            h = run(layer, h)
        return h


class FakeLoopedLM(nn.Module):
    def __init__(self, passes=PASSES):
        super().__init__()
        self.model = LoopedInner(passes)
        text_config = SimpleNamespace(
            num_hidden_layers=LAYERS, loop_layer_index_start=LOOP_START,
            loop_layer_index_end=LOOP_END, loop_num_passes=PASSES,
        )
        self.config = SimpleNamespace(_name_or_path="fake-looped", get_text_config=lambda: text_config)

    @property
    def device(self):
        return torch.device("cpu")

    def forward(self, input_ids, **kw):
        return self.model(input_ids)


def check_looped_result(model, result, trace):
    """result[n, s] = distance(input of position s, output of position s + n - 1)."""
    positions = len(SCHEDULE)
    assert len(trace) == positions
    assert result.shape == (positions + 1, positions)
    assert np.all(result[0] == 0)
    for n in range(1, positions + 1):
        for s in range(positions - n + 1):
            expected = model.distance(trace[s][0], trace[s + n - 1][1])
            assert abs(result[n, s] - expected) < 1e-5, (n, s)
        assert np.all(result[n, positions - n + 1 :] == 0)


def test_looped_model_scores_layer_applications():
    torch.manual_seed(0)
    model = ShortTransformer.from_model(FakeLoopedLM())
    assert model.layer_count == LAYERS and model.is_looped
    assert [i for i, _ in model.layer_schedule] == SCHEDULE
    assert model.layer_labels == LABELS

    text = "abcdef"
    trace = []
    with torch.no_grad():
        model.model(Tok()(text).to("cpu")["input_ids"], trace=trace)
    # the reference run went through the wrapped layers too: start the analysis from clean memory
    model.clear_memory()
    result = model.analyse_layers(dataset=[{"text": text}] * 2, tokenizer=Tok(), key="text", limit=2)
    check_looped_result(model, result, trace)


def test_looped_model_refuses_to_prune():
    model = ShortTransformer.from_model(FakeLoopedLM())
    for call in (lambda: model.prune(start_layer=2, block_size=1),
                 lambda: model.remove_layers(block_size=1, dataset=[{"text": "abcdef"}], tokenizer=Tok())):
        try:
            call()
        except NotImplementedError as e:
            assert "looped" in str(e)
        else:
            raise AssertionError("pruning a looped model was accepted")
    assert len(model.model.language_model.layers) == LAYERS


def test_unexpected_layer_order_is_an_error():
    # the config promises 3 passes, the model runs 4: fail loudly instead of mis-indexing
    model = ShortTransformer.from_model(FakeLoopedLM(passes=PASSES + 1))
    try:
        model.analyse_layers(dataset=[{"text": "abcdef"}], tokenizer=Tok(), key="text")
    except RuntimeError as e:
        assert "position" in str(e)
    else:
        raise AssertionError("a layer order that does not match the schedule was accepted")


def test_from_pretrained_falls_back_to_image_text_to_text():
    from unittest.mock import patch

    looped = FakeLoopedLM()
    with patch("short_transformers.short_transformer.AutoModelForCausalLM") as causal, \
            patch("transformers.AutoModelForImageTextToText") as multimodal:
        causal.from_pretrained.side_effect = ValueError("Unrecognized configuration class")
        multimodal.from_pretrained.return_value = looped
        model = ShortTransformer.from_pretrained("some/vlm")
    multimodal.from_pretrained.assert_called_once_with("some/vlm")
    assert model is looped and model.is_looped


def test_real_inkling2_loop():
    try:
        from transformers import Inkling2LoopConfig, Inkling2LoopForConditionalGeneration
    except ImportError:
        print("skipped: transformers has no Inkling2Loop")
        return

    torch.manual_seed(0)
    cfg = Inkling2LoopConfig(
        text_config=dict(
            vocab_size=64, unpadded_vocab_size=64, hidden_size=32, num_hidden_layers=LAYERS,
            num_attention_heads=2, num_key_value_heads=1, head_dim=16, swa_num_attention_heads=2,
            swa_num_key_value_heads=1, swa_head_dim=16, sliding_window_size=8, d_rel=4, rel_extent=16,
            moe_intermediate_size=16, n_routed_experts=4, num_experts_per_tok=2,
            n_layers_per_attn_res_block=2, loop_layer_index_start=LOOP_START,
            loop_layer_index_end=LOOP_END, loop_num_passes=PASSES, loop_gate_rank=8, eos_token_id=1,
        ),
        vision_config=dict(
            hidden_size=16, num_hidden_layers=1, num_attention_heads=2, text_hidden_size=32,
            projection_hidden_dim=16, posemb_grid_size=4,
        ),
        image_token_id=63,
    )
    model = Inkling2LoopForConditionalGeneration(cfg).eval()
    ids = torch.arange(3, 13)[None]

    # reference inputs/outputs per layer application, from plain torch hooks on an unwrapped model
    trace, pending = [], []
    hooks = []
    for layer in model.model.language_model.layers:
        hooks.append(layer.register_forward_pre_hook(lambda m, a: pending.append(a[0].detach().clone())))
        hooks.append(layer.register_forward_hook(lambda m, a, out: trace.append((pending.pop(), out.detach()))))
    with torch.no_grad():
        model(input_ids=ids)
    for h in hooks:
        h.remove()

    model = ShortTransformer.from_model(model)
    assert model.layer_labels == LABELS

    class IdsTok:
        def __call__(self, text, **kw):
            return SimpleNamespace(to=lambda dev: {"input_ids": ids})

    result = model.analyse_layers(dataset=[{"text": "x"}], tokenizer=IdsTok(), key="text", limit=1)
    check_looped_result(model, result, trace)
    # passes 2 and 3 restart from the pre-loop state, not from the previous pass's output
    assert not torch.allclose(trace[4][0], trace[3][1])
    assert torch.allclose(trace[4][0], trace[1][1])


def test_diagram_with_layer_labels():
    import matplotlib
    matplotlib.use("Agg")
    import tempfile

    from short_transformers.utils import draw_diagram

    positions = len(SCHEDULE)
    r = np.zeros((positions + 1, positions))
    for n in range(1, positions + 1):
        r[n, : positions - n + 1] = np.random.rand(positions - n + 1) + n
    draw_diagram(r, f"{tempfile.mkdtemp()}/d.png", title="looped", labels=LABELS)


def test_scored_blocks_with_layer_labels():
    positions = len(SCHEDULE)
    r = np.ones((positions + 1, positions))
    r[2, 4] = 0.1  # positions 4-5: layers 2 and 3 on the second pass
    md = get_scored_blocks(r, labels=LABELS)
    assert "| 2 | 2p2-3p2 | 0.1|" in md
    assert "| 2 | 4-5 | 0.1|" in get_scored_blocks(r)


if __name__ == "__main__":
    test_import_leaves_root_logger_alone()
    test_ith_token_metrics_on_bf16_and_short_sequences()
    test_plots_close_their_figures()
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
    test_looped_model_scores_layer_applications()
    test_looped_model_refuses_to_prune()
    test_unexpected_layer_order_is_an_error()
    test_from_pretrained_falls_back_to_image_text_to_text()
    test_real_inkling2_loop()
    test_diagram_with_layer_labels()
    test_scored_blocks_with_layer_labels()
    print("ok")
