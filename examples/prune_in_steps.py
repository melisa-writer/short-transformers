from datasets import load_dataset
from short_transformers import ShortTransformer
from short_transformers.graph import draw_diagram
from transformers import AutoModelForCausalLM, AutoTokenizer

# load from path/hf_hub
model_name = "meta-llama/Meta-Llama-3-8B"

model = ShortTransformer.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# or use hf model
# model = ShortTransformer.from_model(hf_model)

dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)

# run inderence on a subset of the datsets
model.analyse_layers(
    dataset=dataset,
    tokenizer=tokenizer,
    key="text",
    limit=1,
    max_length=1000,
)

# after this step, results will be saved to model.memory, which can be saved
model.save_memory("memory.npz")

# and visualized as seaborh graph (see examples in the README.md)
draw_diagram("memory.npz", "memory.png", title="Meta-Llama-3-8B")

# finding optimial block of size 'block_size' to prune
start_layer = model.get_optimal_cut(block_size=5)

# evaluating all possibe block sizes to prune,
# for each block returns score 0-1
# which is averaged over samples distance between input and output to/from a block
block_score = model.get_block_score_stats(return_md=True, threshold=0.3)

# pruning 5-layers block
# this will also clean the `model.memory``
model.cut(start_layer=start_layer, block_size=5)

# or clean the model memory manually
model.clear_memory()

# save the pruned model along with the tokenizer
model.save_pretrained("short_model_path")
