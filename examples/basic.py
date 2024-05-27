from short_transformers import ShortTransformer
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# load from path/hf_hub
model_name = "microsoft/Phi-3-mini-4k-instruct"

model = ShortTransformer.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# or use hf model
# model = ShortTransformer.from_model(hf_model)

dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)

# remove n layers, use hf dataset to find the optimal cut
short_model = model.remove_layers(
    block_size=5, dataset=dataset, tokenizer=tokenizer, key="text"
)

# save as hf model
output_path = "short_model"
short_model.save_pretrained(output_path)

# load again the model using transformers
model = AutoModelForCausalLM.from_pretrained(output_path)
