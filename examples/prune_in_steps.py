from datasets import load_dataset
from short_transformers import ShortTransformer
from short_transformers.utils import (
    draw_diagram,
    get_scored_blocks,
    get_best_pruning_start,
)
from short_transformers.dist import get_angular_distance_ith_token
from transformers import AutoTokenizer

# load from path/hf_hub
model_name = "microsoft/Phi-3-mini-4k-instruct"

model = ShortTransformer.from_pretrained(
    model_name, device_map="auto", trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# optional: change metric
# defaults to: angular_distance_last_token (from the paper)
# calculate distances based on the angular distance of the i=0 token
model.set_metric(get_angular_distance_ith_token(i=0))

# or use hf model
# model = ShortTransformer.from_model(hf_model)

dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)

# calculate distances between inputs/outputs from/to model layers
# results in a triangular numpy array of shape (layer_count, layer_count)
# results[x, y] - averaged distances for block of size x starting at layer y
results = model.analyse_layers(
    dataset=dataset,
    tokenizer=tokenizer,
    key="text",
    limit=100,
    max_length=1000,
)

# draw results (see examples in the README.md)
# by default normalized by rows
draw_diagram(results, "results.png", title="Phi-3-mini-4k-instruct")

# find optimial block of size 'block_size' to prune
start_layer = get_best_pruning_start(results, block_size=5)

# evaluate all possibe block sizes to prune,
# for each block returns score 0-1
# which is averaged over samples distance between input and output to/from a block
block_score = get_scored_blocks(results, return_md=True, threshold=0.3)

# prune 5-layers block
model.prune(start_layer=start_layer, block_size=5)

# save the pruned model along with the tokenizer
model.save_pretrained("short_Phi-3-mini-4k-instruct")
