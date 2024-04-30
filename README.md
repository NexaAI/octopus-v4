<!---
Copyright 2024 The Nexa AI Team. All rights reserved.

Licensed under the Octopus V4 License;
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.nexa4ai.com/use-policy

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

<h1 align="center">Graph of Language Models</h1>

**Let's build this graph together! We need your help. We have tried our best to find the best specialized models, but we can definitely do more with your participation!**

<p align="center">
 <img src="https://storage.googleapis.com/octopus_graph/Octopus4.png" alt="Octopus Logo" width="200">
</p>

This project aims to build the world's largest graph of language models. To our knowledge, it is the first attempt to construct such a graph. Have a look at [our design demo](https://graph.nexa4ai.com/). In this graph, we will integrate many different specialized models and train the respective Octopus models for the edges between the nodes to help distribute and deliver information effectively.

The project is still in its early stages, and we have only included the very initial Octopus model. However, at Nexa AI, we are committed to dedicating significant time and resources to create a powerful graph of AI models.

## Project Scope

The project will mainly focus on the following aspects:

- Identifying the specialized models needed and training these models.
- Constructing the graph consisting of multiple specialized models as nodes.
- Training the Octopus models to connect different nodes efficiently.

The file structure of this GitHub repository is organized as follows:
- `main.py`: This is the primary script for running the Octopus v4 model.
- `build_graph`: Contains methods for constructing and managing the graph of language models. This includes operations such as creating, updating, and deleting nodes and edges.
- `specialized_models`: Here, you'll find the **training code** along with a tutorial on how to prepare your data and train the specialized models. We provide support for two different frameworks: Hugging Face Transformers and PyTorch Lightning, to facilitate your training process. Feel free to raise any issues or questions you encounter during training.



## Environment Setup

We recommend using a Linux environment and assume that you have an NVIDIA GPU when contributing to the project. To set up the project, follow these steps:

```bash
conda create -n octopus4 python=3.10
pip3 install torch torchvision torchaudio
pip3 install transformers datasets accelerate peft
```

Make sure to install PyTorch first, followed by the other packages. Consider to install `torchvision` and `torchaudio` since we will introduce multimodal model in the graph. According to our experience, if you don't install them in one line, there could be some package conflict. Alternatively, you can create a dev environment using our Docker image. For more information on setting up a dev environment, refer to this [YouTube video](https://www.youtube.com/watch?v=0H2miBK_gAk).


## Using the Octopus v4 Model
Our initial v4 model is customized for the MMLU benchmark. However, we plan to support real-world use cases in the future. The Octopus v4 model helps you find the most appropriate model to finish your task and reformats your query so that the worker model can process it effectively. In a graph setup, it knows the best neighbor to choose and how to message from one node to another.

Here's an example of the result for Octopus v4 model:
```text
Query: Tell me the result of derivative of x^3 when x is 2?

<nexa_4>('Determine the derivative of the function f(x) = x^3 at the point where x equals 2, and interpret the result within the context of rate of change and tangent slope.')
<nexa_end>
```

In this use case, `<nexa_4>` is the special token representing the math GPT. The natural math question is converted into a professional math expression to facilitate better understanding by the worker model. To try our model, you can use `python main.py` to run the code to try the Octopus v4 model. 

The respective models used in our experiments are as follows:

###  Model Selection
We leverage the latest Language Large Models for a variety of domains. Below is a summary of the chosen models for each category. In cases where no specialized model exists for a subject, we utilize generic models like Llama3-8b. You may consider to add more content to our table below. Nexa AI will create another leaderboard for the specialized model. 


| **Model**                               | **Category**       | **Subjects**                                                                                                                                                      |
|-----------------------------------------|--------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `jondurbin/bagel-8b-v1.0`               | Biology            | `college_biology`, `high_school_biology`                                                                                                                          |
| `Weyaxi/Einstein-v6.1-Llama3-8B`        | Physics            | `astronomy`, `college_physics`, `conceptual_physics`, `high_school_physics`                                                                                       |
| `meta-llama/Meta-Llama-3-8B-Instruct`   | Business           | `business_ethics`, `management`, `marketing`                                                                                                                      |
| `meta-llama/Meta-Llama-3-8B-Instruct`   | Chemistry          | `college_chemistry`, `high_school_chemistry`                                                                                                                      |
| `abacusai/Llama-3-Smaug-8B`             | Computer Science   | `college_computer_science`, `computer_security`, `high_school_computer_science`, `machine_learning`                                                               |
| `Open-Orca/Mistral-7B-OpenOrca`         | Math               | `abstract_algebra`, `college_mathematics`, `elementary_mathematics`, `high_school_mathematics`, `high_school_statistics`                                          |
| `meta-llama/Meta-Llama-3-8B-Instruct`   | Economics          | `econometrics`, `high_school_macroeconomics`, `high_school_microeconomics`                                                                                       |
| `AdaptLLM/medicine-chat`                | Health             | `anatomy`, `clinical_knowledge`, `college_medicine`, `human_aging`, `medical_genetics`, `nutrition`, `professional_medicine`, `virology`                          |
| `STEM-AI-mtl/phi-2-electrical-engineering` | Engineering     | `electrical_engineering`                                                                                                                                         |
| `meta-llama/Meta-Llama-3-8B-Instruct`   | Philosophy         | `formal_logic`, `logical_fallacies`, `moral_disputes`, `moral_scenarios`, `philosophy`, `world_religions`                                                        |
| `microsoft/Phi-3-mini-128k-instruct`    | Other              | `global_facts`, `miscellaneous`, `professional_accounting`                                                                                                       |
| `meta-llama/Meta-Llama-3-8B-Instruct`   | History            | `high_school_european_history`, `high_school_us_history`, `high_school_world_history`, `prehistory`                                                              |
| `meta-llama/Meta-Llama-3-8B-Instruct`   | Culture            | `human_sexuality`, `sociology`                                                                                                                                   |
| `AdaptLLM/law-chat`                     | Law                | `international_law`, `jurisprudence`, `professional_law`                                                                                                         |
| `meta-llama/Meta-Llama-3-8B-Instruct`   | Psychology         | `high_school_psychology`, `professional_psychology`                                                                                                              |

### MMLU Benchmark Results (5-shot learning)
Here are the comparative MMLU scores for various models tested under a 5-shot learning setup:

| **Model**                         | **MMLU Score** |
|-----------------------------------|----------------|
| Octopus-V4                        | **74.6%**      |
| GPT-3.5                           | 70.0%          |
| Phi-3-mini-128k-instruct          | 68.1%          |
| OpenELM-3B                        | 26.7%          |
| Lamma3-8b-instruct                | 68.4%          |
| Gemma-2b                          | 42.3%          |
| Gemma-7b                          | 64.3%          |



## Train the specialized models
