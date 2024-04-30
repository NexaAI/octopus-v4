from utils import functional_token_mapping, extract_content
from specialized_models_inference import inference_chemistry, inference_math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
torch.random.manual_seed(0)

model_import_mapping = {
    "chemistry_gpt": lambda: inference_chemistry.model(),
    "math_gpt": lambda: inference_math.model()
}

model_inference_mapping = {
    "chemistry_gpt": lambda prompt, pipe, tokenizer: inference_chemistry.inference(prompt, pipe, tokenizer),
    "math_gpt": lambda prompt, pipe, tokenizer: inference_math.inference(prompt, pipe, tokenizer)
}

MAX_TOKEN_LENGTH = 200
NEXA_END_TOKEN_ID = 32041

model = AutoModelForCausalLM.from_pretrained(
    "NexaAIDev/Octopus-v4", 
    device_map="cuda:0", 
    torch_dtype=torch.bfloat16, 
    trust_remote_code=True 
)
tokenizer = AutoTokenizer.from_pretrained("NexaAIDev/octopus-v4-finetuned-v1")

question = "Tell me the result of derivative of x^3 when x is 2?"

inputs = f"<|system|>You are a router. Below is the query from the users, please call the correct function and generate the parameters to call the function.<|end|><|user|>{question}<|end|><|assistant|>"

print('\n============= Below is Octopus-V4 response ==============\n')

# You should consider to use early stopping with <nexa_end> token to accelerate
input_ids = tokenizer(inputs, return_tensors="pt")['input_ids'].to(model.device)

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
print(f'Elapsed time: {end - start:.2f}s')

functional_token, format_argument = extract_content(result)
functional_token = functional_token.strip()
print(f"Functional Token: {functional_token}")
print(f"Format Argument: {format_argument}")

print('\n============= Below is specialized LLM response ==============\n')

if functional_token in functional_token_mapping:
    specialized_model = functional_token_mapping[functional_token]
    pipe, tokenizer = model_import_mapping[specialized_model]()
    response = model_inference_mapping[specialized_model](format_argument, pipe, tokenizer)
    print(response)

