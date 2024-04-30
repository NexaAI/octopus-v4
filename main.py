import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
torch.random.manual_seed(0)

model = AutoModelForCausalLM.from_pretrained(
    "NexaAIDev/Octopus-v4", 
    device_map="cuda:0", 
    torch_dtype=torch.bfloat16, 
    trust_remote_code=True 
)
tokenizer = AutoTokenizer.from_pretrained("NexaAIDev/octopus-v4-finetuned-v1")

question = "Tell me the result of derivative of x^3 when x is 2?"

inputs = f"<|system|>You are a router. Below is the query from the users, please call the correct function and generate the parameters to call the function.<|end|><|user|>{question}<|end|><|assistant|>"

print('\n============= Below is the response ==============\n')

# You should consider to use early stopping with <nexa_end> token to accelerate
input_ids = tokenizer(inputs, return_tensors="pt")['input_ids'].to(model.device)

generated_token_ids = []
start = time.time()

# set a large enough number here to avoid insufficient length
for i in range(200):
    next_token = model(input_ids).logits[:, -1].argmax(-1)
    generated_token_ids.append(next_token.item())
    
    input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=-1)

    # 32041 is the token id of <nexa_end>
    if next_token.item() == 32041:
        break

print(tokenizer.decode(generated_token_ids))
end = time.time()
print(f'Elapsed time: {end - start:.2f}s')
