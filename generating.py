import json
import random
import re
from pathlib import Path
from transformers import AutoProcessor
from PIL import Image
from tqdm import tqdm
import argparse
import sys
import logging
from functools import lru_cache

def main():
    # ---------------------- Configuration ----------------------
    # Define the number of pairs to generate
    NUM_PAIRS = 100000  # 100k pairs

    # Define paths (modify these if necessary)
    META_JSON_PATH = 'evaluation_meta.json'
    INSTRUCTION_JSON_PATH = 'instruction.json'
    OUTPUT_JSON_PATH = 'final.json'
    IMAGE_FOLDER = 'Image_Path'
    REPO = "microsoft/kosmos-2.5"

    # ---------------------- Argument Parsing (Optional) ----------------------
    parser = argparse.ArgumentParser(description='Generate training.json with specified number of pairs.')
    parser.add_argument('--meta', type=str, default=META_JSON_PATH, help='Path to meta JSON file')
    parser.add_argument('--instruction', type=str, default=INSTRUCTION_JSON_PATH, help='Path to instruction JSON file')
    parser.add_argument('--output', type=str, default=OUTPUT_JSON_PATH, help='Path to output JSON file')
    parser.add_argument('--image_folder', type=str, default=IMAGE_FOLDER, help='Folder containing images')
    parser.add_argument('--repo', type=str, default=REPO, help='Processor repository')
    parser.add_argument('--num_pairs', type=int, default=NUM_PAIRS, help='Number of training pairs to generate')
    args = parser.parse_args()

    META_JSON_PATH = args.meta
    INSTRUCTION_JSON_PATH = args.instruction
    OUTPUT_JSON_PATH = args.output
    IMAGE_FOLDER = args.image_folder
    REPO = args.repo
    NUM_PAIRS = args.num_pairs

    # ---------------------- Configure Logging ----------------------
    logging.basicConfig(
        filename='training_data_generation.log',
        level=logging.INFO,
        format='%(asctime)s:%(levelname)s:%(message)s'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    # ---------------------- Validate Paths ----------------------
    if not Path(META_JSON_PATH).exists():
        logging.error(f"Meta JSON file '{META_JSON_PATH}' does not exist.")
        sys.exit(1)
    if not Path(INSTRUCTION_JSON_PATH).exists():
        logging.error(f"Instruction JSON file '{INSTRUCTION_JSON_PATH}' does not exist.")
        sys.exit(1)
    if not Path(IMAGE_FOLDER).exists():
        logging.error(f"Image folder '{IMAGE_FOLDER}' does not exist.")
        sys.exit(1)

    # ---------------------- Load JSON Files ----------------------
    logging.info("Loading JSON files...")
    with open(META_JSON_PATH, 'r', encoding='utf-8') as f:
        meta_data = json.load(f)

    with open(INSTRUCTION_JSON_PATH, 'r', encoding='utf-8') as f:
        instruction_data = json.load(f)

    # ---------------------- Initialize Processor ----------------------
    logging.info(f"Initializing processor from repository '{REPO}'...")
    processor = AutoProcessor.from_pretrained(REPO)

    # ---------------------- Function Definitions ----------------------

    def extract_instruction_details(instruction_text):
        """
        Extracts the prompt and information list from the instruction text.

        Args:
            instruction_text (str): The instruction text.

        Returns:
            tuple: (prompt (str), information (list))
        """
        # Regex patterns to capture prompt and information sections
        prompt_pattern = r"\nINSTRUCTION\n(.*?)\nINFORMATION"
        info_pattern = r"\nINFORMATION\n\[(.*?)\]"

        prompt_match = re.search(prompt_pattern, instruction_text, re.DOTALL)
        info_match = re.search(info_pattern, instruction_text, re.DOTALL)

        if not prompt_match or not info_match:
            raise ValueError("Instruction format is incorrect.")

        prompt = prompt_match.group(1).strip()

        # Safely parse the information list
        info_str = info_match.group(1).strip()
        # Replace single quotes with double quotes for JSON compatibility
        info_str = info_str.replace("'", '"')
        try:
            information = json.loads(f'[{info_str}]')
        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing information list: {e}")

        return prompt, information

    def generate_label(processor, prompt, image, bounding_box, value, is_first):
        """
        Generates a label for a bounding box.

        Args:
            processor (AutoProcessor): The processor instance.
            prompt (str): The prompt text.
            image (PIL.Image): The image.
            bounding_box (dict): The bounding box coordinates.
            value (str): The value inside the bounding box.
            is_first (bool): Flag indicating if this is the first label.

        Returns:
            str: The generated label.
        """
        left_coor = bounding_box["left_top"]
        right_coor = bounding_box["right_bottom"]

        # Process the image to get scaling factors
        inputs = processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        )
        height = inputs.pop("height").item()
        width = inputs.pop("width").item()
        raw_width, raw_height = image.size
        scale_height = raw_height / height
        scale_width = raw_width / width

        # Format the label based on whether it's the first label
        if is_first:
            label_prefix = "<ocr>"
        else:
            label_prefix = ""

        label = (
            f"{label_prefix}<bbox>"
            f"<x_{int(left_coor[0]/scale_width)}>"
            f"<y_{int(left_coor[1]/scale_height)}>"
            f"<x_{int(right_coor[0]/scale_width)}>"
            f"<y_{int(right_coor[1]/scale_height)}>"
            f"></bbox>{value}\n"
        )
        return label

    # ---------------------- Image Caching Mechanism ----------------------
    # To optimize performance, especially with repeated image access, implement a simple caching mechanism.
    # Adjust the cache size based on available memory.

    @lru_cache(maxsize=1000)  # Cache up to 1000 images
    def load_image(image_path):
        """
        Loads and caches an image.

        Args:
            image_path (str): Path to the image.

        Returns:
            PIL.Image: The loaded image in RGB format.
        """
        return Image.open(image_path).convert("RGB")

    # ---------------------- Generate Training Pairs ----------------------
    logging.info(f"Generating {NUM_PAIRS} training pairs...")
    current_key = 0  # Initialize the key counter

    # Open the output file and write the opening bracket
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as outfile:
        outfile.write('[\n')  # Start of JSON array

        batch = []
        batch_size = 100
        is_first_entry = True  # Flag to handle commas
        count = 0
        key_list = list(meta_data.keys())
        with tqdm(total=NUM_PAIRS, desc="Generating Pairs") as pbar:
            while current_key < NUM_PAIRS:
                # Attempt to generate a valid entry
                image_id = random.choice(list(meta_data.keys()))
                # image_id = key_list[count]
                # count += 1
                instruction_key = random.choice(list(instruction_data.keys()))
                instruction_text = instruction_data[instruction_key]

                # Extract prompt and information
                try:
                    prompt, information = extract_instruction_details(instruction_text)
                except ValueError as e:
                    logging.error(f"Skipping instruction '{instruction_key}' due to error: {e}")
                    continue

                # Get the image path

                image_path = Path(IMAGE_FOLDER) / f"{image_id}.png"
                if not image_path.exists():
                    logging.warning(f"Image '{image_path}' does not exist. Skipping.")
                    continue

                # Load the image (with caching)
                try:
                    image = load_image(str(image_path))
                except Exception as e:
                    logging.error(f"Error loading image '{image_path}': {e}. Skipping.")
                    continue

                # Initialize the entry
                entry = {
                    "key": current_key,
                    "image_id": image_id,
                    "prompt": prompt,
                    "information": information,
                    "image_path": str(image_path),
                    "label": ""
                }

                # Process each bounding box in the image
                bounding_boxes = meta_data.get(image_id, [])
                labels = []
                is_first_label = True  # Flag to track the first matching bounding box

                for bbox_info in bounding_boxes:
                    bbox_type = bbox_info.get("type", "").lower()
                    value = bbox_info.get("value", "")
                    coordinates = bbox_info.get("coordinates", {})

                    # Check if the bbox_type is in the information list (case-insensitive)
                    if bbox_type in [info.lower() for info in information]:
                        try:
                            label = generate_label(
                                processor,
                                prompt,
                                image,
                                coordinates,
                                value,
                                is_first_label
                            )
                            labels.append(label)
                            if is_first_label:
                                is_first_label = False  # Subsequent labels won't have <ocr>
                        except Exception as e:
                            logging.error(f"Error generating label for image '{image_id}', bbox type '{bbox_type}': {e}")
                            continue

                # Assign label
                if labels:
                    # Combine all labels into a single string and append the EOS token
                    combined_labels = ''.join(labels) + processor.tokenizer.eos_token
                    entry["label"] = combined_labels
                else:
                    combined_labels = ''.join("No Private Information") + processor.tokenizer.eos_token
                    entry["label"] = combined_labels

                # Append to batch
                batch.append(entry)
                current_key += 1
                pbar.update(1)

                # If batch is full, write to file
                if len(batch) == batch_size:
                    # Convert the batch to JSON string
                    batch_json = json.dumps(batch, ensure_ascii=False, indent=4)
                    # Remove the opening and closing brackets
                    batch_json = batch_json[1:-1] + '\n'

                    if not is_first_entry:
                        # Prefix with a comma to separate JSON objects
                        outfile.write(',\n')
                    else:
                        is_first_entry = False  # Subsequent batches will have commas

                    outfile.write(batch_json)
                    logging.info(f"Wrote batch of {len(batch)} entries to '{OUTPUT_JSON_PATH}'.")
                    # Clear the batch
                    batch = []

            # Handle any remaining entries in the batch
            if batch:
                # Convert the batch to JSON string
                batch_json = json.dumps(batch, ensure_ascii=False, indent=4)
                # Remove the opening and closing brackets
                batch_json = batch_json[1:-1] + '\n'

                if not is_first_entry:
                    # Prefix with a comma to separate JSON objects
                    outfile.write(',\n')
                else:
                    is_first_entry = False  # Subsequent batches will have commas

                outfile.write(batch_json)
                logging.info(f"Wrote final batch of {len(batch)} entries to '{OUTPUT_JSON_PATH}'.")
                # Clear the batch
                batch = []

        # Write the closing bracket of the JSON array
        outfile.write(']\n')
        logging.info(f"Successfully generated {NUM_PAIRS} training pairs in '{OUTPUT_JSON_PATH}'.")

if __name__ == "__main__":
    main()
