# @TODO check torch, transformers installs
# @TODO check model compability
import numpy as np


# result[n, l]: averaged distance for removing the n-layers block starting at layer l,
# valid for l in 0..layer_count-n (see ShortTransformer.Memory)
def get_best_pruning_start(result, block_size: int) -> int:
    layer_count = result.shape[1]
    assert (
        0 < block_size < layer_count
    ), f"Expected `block_size` value between 1 and {layer_count -1}, got {block_size}."
    layer_result = result[block_size, : layer_count - block_size + 1]
    start_layer = int(np.argmin(layer_result))
    return start_layer


def get_scored_blocks(result, return_md=True, threshold=float("inf"), labels=None) -> dict:
    """labels: names for the positions in the markdown table, e.g. model.layer_labels for looped models."""
    layer_count = result.shape[1]
    stats = {}
    for i in range(1, layer_count):
        layer_result = result[i, : layer_count - i + 1]
        start_layer = int(np.argmin(layer_result))
        score = layer_result[start_layer]
        if score <= threshold:
            stats[i] = {"start_layer": start_layer, "score": score}

    if not return_md:
        return stats

    stats_md = "| Block_size | Removed_layers | Score (avg dist)|\n"
    stats_md += "| -------- | ------- | -------- |\n"
    for k, v in stats.items():
        first, last = v["start_layer"], v["start_layer"] + k - 1
        if labels is not None:
            first, last = labels[first], labels[last]
        stats_md += f"| {k} | {first}-{last} | {round(v['score'], 3)}|\n"

    return stats_md
