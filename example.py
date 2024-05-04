from short_transformers import ShortTransformer
from datasets import load_dataset

# load from path/hf_hub
model = ShortTransformer.from_pretrained(model_name)

# or use hf model
# model = ShortTransformer.from_model(hf_model)

dataset = load_dataset(
    "iNeil77/pseudo-mini-pile", "c4_realnews", split="train", streaming=True
)

# remove n layers, use hf dataset to find the optimal cut
short_model = model.remove_layers(
    n=5, dataset=dataset
)  # (n, dataset, key, limit, batch_size, return_outputs, distance)

# save as hf model
output_path = "short_model"
short_mdoel.save_pretrained(output_path)
