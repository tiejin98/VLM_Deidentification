import json
import torch
from transformers import (
    AutoProcessor,
    Kosmos2_5ForConditionalGeneration,
    Trainer,
    TrainingArguments
)
from peft import get_peft_model, LoraConfig, TaskType
from PIL import Image
from datasets import load_dataset, Dataset
from tqdm import tqdm
import argparse
import logging
import sys
from pathlib import Path

def main():
    # ---------------------- Configuration ----------------------
    # Define default paths and parameters
    DEFAULT_META_JSON_PATH = 'training_sample.json'  # Path to your training.json file
    DEFAULT_OUTPUT_MODEL_PATH = '/output/kosmos2.5-finetuned'  # Directory to save the fine-tuned model
    DEFAULT_REPO = "microsoft/kosmos-2.5"  # Pretrained model repository
    DEFAULT_BATCH_SIZE = 2  # Reduced batch size to save memory
    DEFAULT_EPOCHS = 3
    DEFAULT_LEARNING_RATE = 2e-5
    DEFAULT_WEIGHT_DECAY = 0.01
    DEFAULT_LOGGING_DIR = './logs'
    DEFAULT_MAX_LENGTH = 1024  # Adjust based on your GPU memory
    DEFAULT_GRADIENT_ACCUM_STEPS = 8  # To simulate larger batch sizes

    # ---------------------- Argument Parsing ----------------------
    parser = argparse.ArgumentParser(description='Fine-tune Kosmos2.5.')
    parser.add_argument('--meta', type=str, default=DEFAULT_META_JSON_PATH, help='Path to training.json file')
    parser.add_argument('--output_model', type=str, default=DEFAULT_OUTPUT_MODEL_PATH, help='Path to save the fine-tuned model')
    parser.add_argument('--repo', type=str, default=DEFAULT_REPO, help='Pretrained model repository')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=DEFAULT_LEARNING_RATE, help='Learning rate for optimizer')
    parser.add_argument('--weight_decay', type=float, default=DEFAULT_WEIGHT_DECAY, help='Weight decay for optimizer')
    parser.add_argument('--logging_dir', type=str, default=DEFAULT_LOGGING_DIR, help='Directory for logging')
    parser.add_argument('--max_length', type=int, default=DEFAULT_MAX_LENGTH, help='Maximum sequence length for tokenization')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=DEFAULT_GRADIENT_ACCUM_STEPS, help='Gradient accumulation steps')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Device to use for training (cpu or cuda)')
    args = parser.parse_args()

    META_JSON_PATH = args.meta
    OUTPUT_MODEL_PATH = args.output_model
    REPO = args.repo
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LEARNING_RATE = args.learning_rate
    WEIGHT_DECAY = args.weight_decay
    LOGGING_DIR = args.logging_dir
    MAX_LENGTH = args.max_length
    GRAD_ACCUM_STEPS = args.gradient_accumulation_steps
    DEVICE = args.device

    # ---------------------- Configure Logging ----------------------
    logging.basicConfig(
        filename='training_finetuning.log',
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
        logging.error(f"Training JSON file '{META_JSON_PATH}' does not exist.")
        sys.exit(1)

    # ---------------------- Device Configuration ----------------------
    if DEVICE == 'cuda' and not torch.cuda.is_available():
        logging.warning("CUDA is not available. Switching to CPU.")
        DEVICE = 'cpu'

    logging.info(f"Using device: {DEVICE}")

    # ---------------------- Load the Dataset ----------------------
    logging.info("Loading the dataset from training.json...")
    try:
        dataset = load_dataset('json', data_files=META_JSON_PATH, split='train')
        logging.info(f"Loaded dataset with {len(dataset)} entries.")
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        sys.exit(1)

    # ---------------------- Split the Dataset ----------------------
    logging.info("Splitting the dataset into training and evaluation sets (95k train, 5k eval)...")
    try:
        split_ratio = 0.65  # 5% for evaluation
        split_datasets = dataset.train_test_split(test_size=split_ratio, seed=42)
        train_dataset = split_datasets['train']
        eval_dataset = split_datasets['test']
        logging.info(f"Training set: {len(train_dataset)} examples")
        logging.info(f"Evaluation set: {len(eval_dataset)} examples")
    except Exception as e:
        logging.error(f"Failed to split dataset: {e}")
        sys.exit(1)
    print(train_dataset[0])
    # ---------------------- Initialize Processor ----------------------
    logging.info(f"Initializing processor from repository '{REPO}'...")
    try:
        processor = AutoProcessor.from_pretrained("microsoft/kosmos-2.5")
    except Exception as e:
        logging.error(f"Failed to initialize processor: {e}")
        sys.exit(1)

    def preprocess_function(examples):
        prompts = examples['prompt']
        labels = examples['label']
        full_text = [f"{p} {l}" for p, l in zip(prompts, labels)]
        resized_images = []

        for img_path in examples['image_path']:
            try:
                image = Image.open(img_path).convert("RGB")
                resized_images.append(image)
            except Exception as e:
                logging.error(f"Error loading or resizing image '{img_path}': {e}")
                resized_images.append(Image.new('RGB', (512, 512), color=(255, 255, 255)))

        encoding = processor(
            text=full_text,
            images=resized_images,
            return_tensors="pt",
            # Remove padding and truncation arguments
        )
        # model cannot see height or width, must pop out
        height, width = encoding.pop("height"), encoding.pop("width")

        prompt_lengths = []
        for p in prompts:
            prompt_encoding = processor.tokenizer(p, return_tensors="pt")
            prompt_length = prompt_encoding['input_ids'].shape[1]
            prompt_lengths.append(prompt_length)

        labels_tensor = encoding['input_ids'].clone()

        for i, prompt_length in enumerate(prompt_lengths):
            if prompt_length < labels_tensor.size(1):
                labels_tensor[i, :prompt_length] = -100

        encoding['labels'] = labels_tensor

        return encoding

    # ---------------------- Apply On-the-Fly Transformation ----------------------
    logging.info("Setting on-the-fly transformation for the training dataset...")
    try:
        train_dataset.set_transform(preprocess_function)
        logging.info("Transformation set successfully for the training dataset.")
    except Exception as e:
        logging.error(f"Failed to set transformation for training dataset: {e}")
        sys.exit(1)

    logging.info("Setting on-the-fly transformation for the evaluation dataset...")
    try:
        eval_dataset.set_transform(preprocess_function)
        logging.info("Transformation set successfully for the evaluation dataset.")
    except Exception as e:
        logging.error(f"Failed to set transformation for evaluation dataset: {e}")
        sys.exit(1)


    logging.info("Loading the pretrained Kosmos2.5 model...")
    try:
        model = Kosmos2_5ForConditionalGeneration.from_pretrained(
            REPO,
            torch_dtype=torch.bfloat16 if DEVICE == 'cuda' else torch.float32,
            cache_dir="model/",
            trust_remote_code=True  # Necessary if the model uses custom code
        )
    except Exception as e:
        logging.error(f"Failed to load the model: {e}")
        sys.exit(1)


    for param in model.parameters():
        param.requires_grad = True


    # ---------------------- Define Training Arguments ----------------------
    # Determine whether to disable CUDA based on device argument
    no_cuda = False if DEVICE == 'cuda' else True

    training_args = TrainingArguments(
        output_dir=OUTPUT_MODEL_PATH,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,  # To simulate larger batch sizes
        # evaluation_strategy="steps",  # evaluate based on steps instead of epochs
        # eval_steps=500,               # evaluate every 500 steps (or any interval you prefer)
        save_strategy="steps",        # saving based on steps as well, if desired
        save_steps=2000,               # match with eval_steps or set independently
        logging_dir=LOGGING_DIR,
        logging_steps=100,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        save_total_limit=3,  # Keep only the last 3 checkpoints
        bf16=True if DEVICE == 'cuda' else False,  # Enable mixed precision if CUDA is available
        no_cuda=no_cuda,  # Disable CUDA if device is set to 'cpu'
        report_to="none",  # Disable reporting to Weights & Biases or other services
        remove_unused_columns=False,
    )

    # ---------------------- Initialize Trainer ----------------------
    logging.info("Initializing the Trainer...")
    try:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=processor,
        )
    except Exception as e:
        logging.error(f"Failed to initialize Trainer: {e}")
        sys.exit(1)

    # ---------------------- Start Training ----------------------
    logging.info("Starting training...")
    try:
        trainer.train()
    except KeyboardInterrupt:
        logging.warning("Training interrupted by user.")
        trainer.save_model(OUTPUT_MODEL_PATH)
        logging.info(f"Model saved to '{OUTPUT_MODEL_PATH}' after interruption.")
        sys.exit(0)
    logging.info("Training completed.")

    logging.info(f"Saving the fine-tuned model to '{OUTPUT_MODEL_PATH}'...")
    try:
        model.save_pretrained(OUTPUT_MODEL_PATH)
        processor.save_pretrained(OUTPUT_MODEL_PATH)
        logging.info("Model and processor saved successfully.")
    except Exception as e:
        logging.error(f"Failed to save the model: {e}")
        sys.exit(1)

    # ---------------------- End of Script ----------------------

if __name__ == "__main__":
    main()
