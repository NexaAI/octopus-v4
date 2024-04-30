# Part of code below is adapted from the microsoft/phi3
# https://huggingface.co/microsoft/Phi-3-mini-128k-instruct/blob/main/sample_finetune.py
import sys
import logging
import datasets
from datasets import load_dataset
from peft import LoraConfig
import torch
import transformers
from trl import SFTTrainer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
import os

logger = logging.getLogger(__name__)

# set your key parameters to support the training
# for base model, make sure a compatible tokenizer is placed inside the directory
# as well. Just use tokenizer.push_to_hub("same path as model")
PARAMS = {
    "base_model": "google/gemma-2b",  # You can switch to another model
    "dataset_name": "microsoft/orca-math-word-problems-200k",
    "WANDB_DISABLED": "true",  # set as false if you want to use wandb
}

os.environ["WANDB_DISABLED"] = PARAMS["WANDB_DISABLED"]

# Set the training configuration
# Refer to HF alignment-handbook/scripts/run_sft.py for more usage about trl
training_config = {
    "bf16": True,
    "do_eval": False,
    "gradient_accumulation_steps": 1,
    "gradient_checkpointing": True,
    "gradient_checkpointing_kwargs": {"use_reentrant": False},
    "learning_rate": 5.0e-06,
    "log_level": "info",
    "logging_steps": 20,
    "logging_strategy": "steps",
    "lr_scheduler_type": "cosine",
    "max_steps": -1,
    "num_train_epochs": 2,
    "output_dir": "./checkpoints",
    "overwrite_output_dir": True,
    "per_device_eval_batch_size": 4,
    "per_device_train_batch_size": 4,
    "remove_unused_columns": True,
    "save_steps": 100,
    "save_total_limit": 1,
    "seed": 0,
    "warmup_ratio": 0.2,
}

# Consider to change lora setup to save your memory
peft_config = {
    "bias": "none",
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "modules_to_save": None,
    "r": 16,
    "task_type": "CAUSAL_LM",
    "target_modules": "all-linear",
}

train_conf = TrainingArguments(**training_config)
peft_conf = LoraConfig(**peft_config)

# set the logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log_level = train_conf.get_process_log_level()
logger.setLevel(log_level)
datasets.utils.logging.set_verbosity(log_level)
transformers.utils.logging.set_verbosity(log_level)
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()

# Log on each process a small summary
logger.warning(
    f"Process rank: {train_conf.local_rank}, device: {train_conf.device}, n_gpu: {train_conf.n_gpu}"
    + f" distributed training: {bool(train_conf.local_rank != -1)}, 16-bits training: {train_conf.fp16}"
)
logger.info(f"Training/evaluation parameters {train_conf}")
logger.info(f"PEFT parameters {peft_conf}")

model_kwargs = dict(
    use_cache=False,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    device_map=None,
)

model = AutoModelForCausalLM.from_pretrained(PARAMS["base_model"], **model_kwargs)
tokenizer = AutoTokenizer.from_pretrained(PARAMS["base_model"])
tokenizer.model_max_length = 2048
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


# TRL is a highly wrapped library,
# we should use text field to store the final training data
# and other fields would be removed. You can use process dataset to assemble the data
def process_dataset(example):
    question = example["question"]
    answer = example["answer"]
    example["text"] = f"Question:{question},  Answer:{answer}"
    return example


# --------------------------------------------------------------------- #
#  To help you test the code, we only select 10000 samples. In real     #
#  use case, you should include all data. And the train/val data should #
#  be splitted.                                                         #
# --------------------------------------------------------------------- #
raw_dataset = load_dataset(PARAMS["dataset_name"], split="train").select(range(10000))
raw_dataset = raw_dataset.train_test_split(test_size=128, seed=42)

train_dataset = raw_dataset["train"]
test_dataset = raw_dataset["test"]
column_names = list(train_dataset.features)

processed_train_dataset = train_dataset.map(
    process_dataset,
    num_proc=4,
    remove_columns=column_names,
    desc="Preprocessing the dataset",
)

processed_test_dataset = test_dataset.map(
    process_dataset,
    num_proc=4,
    remove_columns=column_names,
    desc="Preprocessing the dataset",
)

# training
trainer = SFTTrainer(
    model=model,
    args=train_conf,
    peft_config=peft_conf,
    train_dataset=processed_train_dataset,
    eval_dataset=processed_test_dataset,
    max_seq_length=2048,
    dataset_text_field="text",  # select text as the data field
    tokenizer=tokenizer,
    packing=True,
)

train_result = trainer.train()
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()

# Save the trained model
trainer.save_model(train_conf.output_dir)
