from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "deepseek-ai/deepseek-coder-6.7b-instruct"

def model(accelerator, model_name_or_id=model_id):
    # Setting up the model with necessary parameters
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_id,
        device_map={"": accelerator.process_index} if accelerator else "auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_id)
    return model, tokenizer

def prompt_format(user_query, tokenizer):
    # Define a chat template
    chat_template = [
        {"role": "system", "content": "You are an expert coder, skilled in various programming languages, algorithms, and software development practices."},
        {"role": "user", "content": user_query}
    ]
    # Prepare the chat for the model (tokenize the chat if necessary)
    prepared_chat = tokenizer.apply_chat_template(chat_template, tokenize=False)   
    return prepared_chat

def inference(prompt, pipe, tokenizer):
    formatted_prompt = prompt_format(prompt, tokenizer)
    prompt_tokenized = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
    output_tokenized = pipe.generate(
        **prompt_tokenized,
        max_length=2048,
        temperature=1,
        max_new_tokens=512, 
        do_sample=False, 
        top_k=50, 
        top_p=0.95, 
        num_return_sequences=1, 
        eos_token_id=tokenizer.eos_token_id
    )
    # Decode generated tokens to string
    answer = tokenizer.decode(output_tokenized[0], skip_special_tokens=True)
    return answer[len(formatted_prompt):]

if __name__ == "__main__":
    prompt = "Write a Python function to merge two sorted lists into a single sorted list."
    pipe, tokenizer = model(None, )
    response = inference(prompt, pipe, tokenizer)
    print(response)
