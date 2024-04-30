# Overview

In this folder, we introduce the framework to train the specialized models. We currently focus on the "small" language models that are less than 10B parameters. 

Some general tips for the training code:
1. It will trigger the `wandb` logging system. We recommend to register a `wandb` account;
2. All accelerate capability can be used here as well. Consider to set the accelerate config for your problem, and use `accelerate launch run.py` to start the training. For the setup of the accelerate, please refer to the [accelerate documentation](https://huggingface.co/docs/accelerate/index). You can use `accelerate` to adjust the mode of distributed training. Here is a [short description of different type of distributed learning](https://alexchen4ai.github.io/blog/notes/Large%20Language%20Model/llm_train.html);
3. Please file any issus you encounter during the training process. We will try to help you as soon as possible.


## SFT training
When we run the model, use `accelerate launch --num_processes=$NUM_GPUS train_script.py` to start training.

Make sure to include the flag `--num_processes=$NUM_GPUS` to specify the number of GPUs to use. Otherwise, some environment may have issues with the GPU allocation. Since language models are not distinctly seperated into completion and chat model. We here differentiate them as well. The chat model can be trained after we introduce some special tokens to indicate the `<system>`, `<user>` and `<assistant>`. 

To help use the training code more easily, we split the two different training scripts. 

### Completion model trainer
Use the `sft_completion.py` script, and the command is `accelerate launch --num_processes=$NUM_GPUS sft_completion.py`. Consider to change the value of `PARAMS` to customize your training setup. 

### Chat model trainer
If you want to train a chat model, you may need a instructed model with tokens for the chat config. Also, the dataset used is different. We have prepared a script called `sft_chat.py`, and you also need to use the command `accelerate launch --num_processes=$NUM_GPUS sft_chat.py`. Consider to change the value of `PARAMS` to customize your training setup.

Make sure you use the correct tokenizer that has the matching special token for chatting.

### DPO
