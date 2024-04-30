# Overview

In this folder, we introduce the framework to train the specialized models. We currently focus on the "small" language models that are less than 10B parameters. 

Some general tips for the training code:
1. It will trigger the `wandb` logging system. We recommend to register a `wandb` account;
2. All accelerate capability can be used here as well. Consider to set the accelerate config for your problem, and use `accelerate launch run.py` to start the training. For the setup of the accelerate, please refer to the [accelerate documentation](https://huggingface.co/docs/accelerate/index). You can use `accelerate` to adjust the mode of distributed training. Here is a [short description of different type of distributed learning](https://alexchen4ai.github.io/blog/notes/Large%20Language%20Model/llm_train.html);
3. Please file any issus you encounter during the training process. We will try to help you as soon as possible.


## SFT training
