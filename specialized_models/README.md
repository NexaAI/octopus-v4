# Training Framework for Specialized Models

This guide provides an overview of how to train "small" language models with fewer than 10 billion parameters. Before starting, please take note of these key points:

1. **Logging with Weights & Biases**: The training process utilizes the `wandb` logging system. It's recommended to [register a `wandb` account](https://wandb.ai/site) if you haven't done so already.
2. **Utilizing Accelerate**: All acceleration capabilities can be leveraged here. It's advisable to configure the `accelerate` settings according to your specific requirements. Use the command `accelerate launch run.py` to initiate training. For setup instructions, refer to the [Accelerate documentation](https://huggingface.co/docs/accelerate/index). Additionally, here's a [brief on different types of distributed training](https://alexchen4ai.github.io/blog/notes/Large%20Language%20Model/llm_train.html) that you may find useful.
3. **Issue Reporting**: Should you encounter any problems during training, please file an issue on this repository. We aim to provide assistance promptly.

## Specific Training Instructions

### SFT Training
To train your model, execute the following command, ensuring you replace `$NUM_GPUS` with the actual number of GPUs available:

```bash
accelerate launch --num_processes=$NUM_GPUS train_script.py
```

This will handle GPU allocation and prevent potential issues. Our framework distinguishes between completion and chat models by using specific training scripts and special tokens `<system>`, `<user>`, and `<assistant>`.

#### Completion Model Training
For the completion model:
```bash
accelerate launch --num_processes=$NUM_GPUS sft_completion.py
```
Customize your training by modifying the `PARAMS` in the script as necessary.

#### Chat Model Training
For the chat model, which requires a tokenizer with special tokens configured for chatting:
```bash
accelerate launch --num_processes=$NUM_GPUS sft_chat.py
```
Again, adjust `PARAMS` to tailor the training setup to your needs.

