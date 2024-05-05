# :scissors: Short Transformers

- [Unofficial] Pytorch implementation of layer pruning proposed in [The Unreasonable Ineffectiveness of the Deeper Layers](https://arxiv.org/pdf/2403.17887.pdf).
- The repository reproduces and extends original methods by offering different layer pruning criteria.

<p align="center">
<img src="./docs/merged.png" align="center" alt="Normalized angular distance from initial layer l (x-axis) with block size n (y-axis)." height='250'/>
</p>


[![pypi Version](https://img.shields.io/pypi/v/short-transformers.svg?style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/short-transformers/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## Installation:
```sh
pip install short-transformers
```

Required additional dependencies: `transformers`, `datasets`.

## Quickstart:
```python
from short_transformers import ShortTransformer
from datasets import load_dataset

# load from path/hf_hub
model = ShortTransformer.from_pretrained(model_name)

# or use hf model
# model = ShortTransformer.from_model(hf_model)

# load hf dataset
dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)

# remove 5 layers, use the dataset to find the least important layers to remove
short_model = model.remove_layers(block_size=5, dataset=dataset, limit=1000)

# continue training to heal after the cut
# ...

# save as hf model
short_mdoel.save_pretrained(output_path)
```

Both `short_model` and the saved model are fully compatible with transformers. See `examples/basic.py` for a complete working example.

## Pruning in steps:

Pruning can composed step-by-step and customized:

1. Analyze model layers:
```python
from datasets import load_dataset
from short_transformers import ShortTransformer
from short_transformers.utils import (
    draw_diagram,
    get_scored_blocks,
    get_best_pruning_start,
)
# load from path/hf_hub
model_name = "meta-llama/Meta-Llama-3-8B"

model = ShortTransformer.from_pretrained(model_name, device_map="auto")

dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)

# calculate distances between inputs/outputs from/to model layers
# results in a triangular numpy array of shape (layer_count, layer_count)
# results[x, y] - averaged distances for block of size x starting at layer y
results = model.analyse_layers(
    dataset=dataset,
    key="text",
    limit=100,
    max_length=1000,
)

# draw results
# diagrams style matches the style of original article
# "The Unreasonable Ineffectiveness of the Deeper Layers"
draw_diagram(results, "results.png", title="Meta-Llama-3-8B")
```

Example output:
<p align="center">
<img src="./docs/Meta-Llama-3-8B.png" align="center" width='300'/>
</p>

2. Find optimal `block_size` and `start_layer`:
```python
# find optimial block of size 'block_size' to prune
start_layer = get_best_pruning_start(results, block_size=5)

# evaluate all possibe block sizes to prune,
# for each block returns score 0-1
# which is averaged over samples distance between input and output to/from a block
block_score = get_scored_blocks(results, return_md=True, threshold=0.3)
```

Example output:

| Block_size | Removed_layers | Score (avg dist)|
| -------- | ------- | -------- |
| 1 | 25-25 | 0.123|
| 2 | 24-25 | 0.155|
| 3 | 25-27 | 0.181|
| 4 | 24-27 | 0.204|
| 5 | 23-27 | 0.226|
| 6 | 22-27 | 0.248|
| 7 | 22-28 | 0.268|
| 8 | 20-27 | 0.291|


3. Pruning layers:
```python
# prune 5-layers block
model.prune(start_layer=start_layer, block_size=5)

# save the pruned model
model.save_pretrained("model_output_dir")
```

See `example/prune_in_steps.py` for a complete working example.


## Supported pruning methods:
- based on layer input/output distances:
    - angular distance of the last token (original)
    - averaged angular distances of all tokens

- todo: based on layer linear replacement trining loss

## Citing

If you use Short Transformers in your research, please cite with the following BibText

```bibtext
@misc{russak2024shorttransformers,
    title  = {ShortTransformers, optimal layer pruning tools},
    author = {Melisa Russak},
    url    = {https://github.com/melisa/short-transformers},
    year   = {2024}
}
```
```bibtext
@misc{gromov2024unreasonable,
      title={The Unreasonable Ineffectiveness of the Deeper Layers}, 
      author={Andrey Gromov and Kushal Tirumala and Hassan Shapourian and Paolo Glorioso and Daniel A. Roberts},
      year={2024},
      eprint={2403.17887},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```