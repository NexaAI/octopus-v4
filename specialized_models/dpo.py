# imports
import os
import sys
import logging
import torch
from dataclasses import dataclass
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    set_seed,
)
from trl import DPOTrainer
from accelerate import Accelerator

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# Define key parameters and setup, consider to change the parameters to fit your needs
PARAMS = {
    "model_name_or_path": "google/gemma-2b-it",
    "output_dir": "./dpo_checkpoints",
    "max_steps": 1000,
    "train_batch_size": 4,
    "eval_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "learning_rate": 1e-6,
    "lr_scheduler_type": "cosine",
    "warmup_steps": 100,
    "logging_steps": 10,
    "save_steps": 100,
    "eval_steps": 100,
    "model_dtype": "bfloat16",
    "beta": 0.1,
}

os.environ["WANDB_DISABLED"] = "true"  # Disable wandb logging

# Load the model and tokenizer
torch_dtype = getattr(torch, PARAMS["model_dtype"])
model = AutoModelForCausalLM.from_pretrained(
    PARAMS["model_name_or_path"],
    torch_dtype=torch_dtype,
    device_map="auto",
)
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(PARAMS["model_name_or_path"])
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


# Load dataset, make sure your dataset only has three columns, 
# "question", "chosen", "rejected"
# The dataset alexchen4ai/orca_dpo_pairs is adapted from Intel/orca_dpo_pairs
raw_dataset = load_dataset("alexchen4ai/orca_dpo_pairs", split="train").train_test_split(0.1)
train_dataset = raw_dataset["train"]
eval_dataset = raw_dataset["test"]



# Set training arguments
train_conf = TrainingArguments(
    output_dir=PARAMS["output_dir"],
    num_train_epochs=2,
    per_device_train_batch_size=PARAMS["train_batch_size"],
    per_device_eval_batch_size=PARAMS["eval_batch_size"],
    max_steps=PARAMS["max_steps"],
    logging_steps=PARAMS["logging_steps"],
    save_steps=PARAMS["save_steps"],
    evaluation_strategy="steps",
    eval_steps=PARAMS["eval_steps"],
    learning_rate=PARAMS["learning_rate"],
    warmup_steps=PARAMS["warmup_steps"],
    lr_scheduler_type=PARAMS["lr_scheduler_type"],
    optim="adamw_torch",
)

# Initialize the trainer
dpo_trainer = DPOTrainer(
    model=model,
    args=train_conf,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    beta=PARAMS["beta"],
)

# Start training
train_result = dpo_trainer.train()
dpo_trainer.save_model(PARAMS["output_dir"])
