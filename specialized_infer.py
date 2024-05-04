from utils import functional_token_mapping, extract_content
from specialized_models_inference import (
    inference_biology,
    inference_business,
    inference_chemistry,
    inference_computer_science,
    inference_math,
    inference_physics,
    inference_electrical_engineering,
    inference_history,
    inference_philosophy,
    inference_law,
    inference_politics,
    inference_culture,
    inference_economics,
    inference_geography,
    inference_psychology,
    inference_health,
    inference_general,
)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

torch.random.manual_seed(0)

model_import_mapping = {
    "physics_gpt": lambda: inference_physics.model(),
    "chemistry_gpt": lambda: inference_chemistry.model(),
    "biology_gpt": lambda: inference_biology.model(),
    "computer_science_gpt": lambda: inference_computer_science.model(),
    "math_gpt": lambda: inference_math.model(),
    "business_gpt": lambda: inference_business.model(),
    "electrical_engineering_gpt": lambda: inference_electrical_engineering.model(),
    "history_gpt": lambda: inference_history.model(),
    "philosophy_gpt": lambda: inference_philosophy.model(),
    "law_gpt": lambda: inference_law.model(),
    "politics_gpt": lambda: inference_politics.model(),
    "culture_gpt": lambda: inference_culture.model(),
    "economics_gpt": lambda: inference_economics.model(),
    "geography_gpt": lambda: inference_geography.model(),
    "psychology_gpt": lambda: inference_psychology.model(),
    "health_gpt": lambda: inference_health.model(),
    "general_gpt": lambda: inference_general.model(),
}

model_inference_mapping = {
    "chemistry_gpt": lambda prompt, pipe, tokenizer: inference_chemistry.inference(
        prompt, pipe, tokenizer
    ),
    "math_gpt": lambda prompt, pipe, tokenizer: inference_math.inference(
        prompt, pipe, tokenizer
    ),
    "computer_science_gpt": lambda prompt, pipe, tokenizer: inference_computer_science.inference(
        prompt, pipe, tokenizer
    ),
    "biology_gpt": lambda prompt, pipe, tokenizer: inference_biology.inference(
        prompt, pipe, tokenizer
    ),
    "business_gpt": lambda prompt, pipe, tokenizer: inference_business.inference(
        prompt, pipe, tokenizer
    ),
    "physics_gpt": lambda prompt, pipe, tokenizer: inference_physics.inference(
        prompt, pipe, tokenizer
    ),
    "electrical_engineering_gpt": lambda prompt, pipe, tokenizer: inference_electrical_engineering.inference(
        prompt, pipe, tokenizer
    ),
    "history_gpt": lambda prompt, pipe, tokenizer: inference_history.inference(
        prompt, pipe, tokenizer
    ),
    "philosophy_gpt": lambda prompt, pipe, tokenizer: inference_philosophy.inference(
        prompt, pipe, tokenizer
    ),
    "law_gpt": lambda prompt, pipe, tokenizer: inference_law.inference(
        prompt, pipe, tokenizer
    ),
    "politics_gpt": lambda prompt, pipe, tokenizer: inference_politics.inference(
        prompt, pipe, tokenizer
    ),
    "culture_gpt": lambda prompt, pipe, tokenizer: inference_culture.inference(
        prompt, pipe, tokenizer
    ),
    "economics_gpt": lambda prompt, pipe, tokenizer: inference_economics.inference(
        prompt, pipe, tokenizer
    ),
    "geography_gpt": lambda prompt, pipe, tokenizer: inference_geography.inference(
        prompt, pipe, tokenizer
    ),
    "psychology_gpt": lambda prompt, pipe, tokenizer: inference_psychology.inference(
        prompt, pipe, tokenizer
    ),
    "health_gpt": lambda prompt, pipe, tokenizer: inference_health.inference(
        prompt, pipe, tokenizer
    ),
    
}

MAX_TOKEN_LENGTH = 200
NEXA_END_TOKEN_ID = 32041

model = AutoModelForCausalLM.from_pretrained(
    "NexaAIDev/Octopus-v4",
    device_map="cuda:0",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("NexaAIDev/octopus-v4-finetuned-v1")

question = "Tell me the result of derivative of x^3 when x is 2?"

inputs = f"<|system|>You are a router. Below is the query from the users, please call the correct function and generate the parameters to call the function.<|end|><|user|>{question}<|end|><|assistant|>"

print("\n============= Below is Octopus-V4 response ==============\n")

# You should consider to use early stopping with <nexa_end> token to accelerate
input_ids = tokenizer(inputs, return_tensors="pt")["input_ids"].to(model.device)

generated_token_ids = []
start = time.time()

# set a large enough number here to avoid insufficient length
for i in range(MAX_TOKEN_LENGTH):
    next_token = model(input_ids).logits[:, -1].argmax(-1)
    generated_token_ids.append(next_token.item())
    input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=-1)
    if next_token.item() == NEXA_END_TOKEN_ID:
        break

result = tokenizer.decode(generated_token_ids)
end = time.time()
print(f"{result}")
print(f"Elapsed time: {end - start:.2f}s")

functional_token, format_argument = extract_content(result)
functional_token = functional_token.strip()
print(f"Functional Token: {functional_token}")
print(f"Format Argument: {format_argument}")

print("\n============= Below is specialized LLM response ==============\n")

if functional_token in functional_token_mapping:
    specialized_model = functional_token_mapping[functional_token]
    pipe, tokenizer = model_import_mapping[specialized_model]()
    response = model_inference_mapping[specialized_model](
        format_argument, pipe, tokenizer
    )
    print(response)
