# @TODO check torch, transformers installs
# @TODO check model compability
import numpy as np


def get_best_pruning_start(result, block_size: int) -> int:
    layer_count = result.shape[0]
    assert (
        block_size < layer_count and block_size > 0
    ), f"Expected `block_size` value between 1 and {layer_count -1}, got {block_size}."
    layer_result = result[block_size, : layer_count - block_size]
    start_layer = np.argmin(layer_result)
    return start_layer


def get_scored_blocks(result, return_md=True, threshold=float("inf")) -> dict:
    layer_count = result.shape[0]
    stats = {}
    for i in range(1, layer_count):
        layer_result = result[i, : layer_count - i]
        start_layer = np.argmin(layer_result)
        score = layer_result[start_layer]
        if score <= threshold:
            stats[i] = {"start_layer": start_layer, "score": score}

    if not return_md:
        return stats

    stats_md = "| Block_size | Removed_layers | Score (avg dist)|\n"
    stats_md += "| -------- | ------- | -------- |\n"
    for k, v in stats.items():
        stats_md += f"| {k} | {v['start_layer']}-{v['start_layer']+k-1} | {round(v['score'], 3)}|\n"

    return stats_md
