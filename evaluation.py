from transformers import AutoProcessor, Kosmos2_5ForConditionalGeneration
import re
import torch
import requests
from PIL import Image, ImageDraw
import json
from tqdm import tqdm

# Set device and dtype
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16  # Change to torch.float16 if needed

# Load the base model
repo = "/output/kosmos2.5-finetuned"
model = Kosmos2_5ForConditionalGeneration.from_pretrained(
    repo,
    torch_dtype=dtype,
    # attn_implementation="flash_attention_2",
)
model = model.to(device)

# Load the processor
processor = AutoProcessor.from_pretrained("microsoft/kosmos-2.5")


# Wrap the base model with PeftModel and load the adapters
model.eval()  # Set model to evaluation mode

json_path = "evaluation_Base.json"
save_path = "eva_result.json"
# Create the prompt
final_res = {}

with open(json_path, 'r', encoding='utf-8') as f:
    meta_data = json.load(f)
for sample in tqdm(meta_data):
    key = sample['key']
    prompt = sample['prompt']
    image_path = sample['image_path']
    image = Image.open(f'{image_path}')
    inputs = processor(
    text=prompt,
    images=image,
    return_tensors="pt",
    )

    inputs = {k: v.to(device) if v is not None else None for k, v in inputs.items()}
    height, width = inputs.pop("height"), inputs.pop("width")
    raw_width, raw_height = image.size
    scale_height = raw_height / height
    scale_width = raw_width / width
    # Cast input tensors to the model's dtype
    inputs["flattened_patches"] = inputs["flattened_patches"].to(dtype)
    length = inputs['input_ids'].shape[1]
    # Generate output
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=150,
    )

    # Decode the generated tokens
    generated_text = processor.batch_decode(generated_ids[:,length:])
    final_res[key] = generated_text

with open(save_path, 'w') as json_file:
    json.dump(final_res, json_file, indent=4)
