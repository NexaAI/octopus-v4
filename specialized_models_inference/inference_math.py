from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

model_id = "Open-Orca/Mistral-7B-OpenOrca"

def model(accelerator=None, model_name_or_id = model_id):
    # Setting up the model with necessary parameters
    model = AutoModelForCausalLM.from_pretrained(model_name_or_id,
                                                              device_map={"": accelerator.process_index} if accelerator else "auto",
                                                              torch_dtype="auto", 
                                                              trust_remote_code=True
                                                              )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_id)
    return model, tokenizer


def prompt_format(user_query):
    llama_prompt_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{{You are expert to solve math problems.}}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    return llama_prompt_template.format(user_query=user_query)


def inference(prompt, pipe, tokenizer):
    formatted_prompt = prompt_format(user_query=prompt)
    prompt_tokenized = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
    output_tokenized = pipe.generate(**prompt_tokenized, 
                             max_length=2048, 
                             num_return_sequences=1,
                             no_repeat_ngram_size=2,
                             temperature=0.7)
    # Decode generated tokens to string
    answer = tokenizer.decode(output_tokenized[0], skip_special_tokens=True)
    return answer[len(formatted_prompt):]

if __name__ == "__main__":
    prompt = "Tell me the result of derivative of x^3 when x is 2?"
    pipe, tokenizer = model(None)
    response = inference(prompt, pipe, tokenizer)
    print(response)