# Overview

In this folder, we introduce the framework to train the specialized models. We currently focus on the "small" language models that are less than 10B parameters. 

Some general tips for the training code:
1. It will trigger the `wandb` logging system. We recommend to register a `wandb` account.
2. All accelerate capability can be used here as well. Consider to set the accelerate config for your problem, and use `accelerate launch run.py` to start the training.


## SFT training
