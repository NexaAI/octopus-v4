from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Open-Orca/Mistral-7B-OpenOrca"

# Function to load the model and tokenizer
def model(accelerator=None, model_name_or_id=model_id):
    # Setting up the model with necessary parameters
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_id,
        device_map={"": accelerator.process_index} if accelerator else "auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_id)
    return model, tokenizer

def prompt_format(user_query):
    # Defines the format for the prompt
    prompt_template = """system
    {system}
    user
    {user_query}
    assistant
    """
    return prompt_template.format(system="You are an expert in solving math problems.", user_query=user_query)

def inference(prompt, model, tokenizer):
    # Format and tokenize the prompt
    formatted_prompt = prompt_format(user_query=prompt)
    prompt_tokenized = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
    # Generate the response using the model
    output_tokenized = model.generate(
        **prompt_tokenized,
        max_length=2048,
        num_return_sequences=1,
        temperature=1
    )
    # Decode generated tokens to string
    answer = tokenizer.decode(output_tokenized[0], skip_special_tokens=True)
    return answer[len(formatted_prompt):]

if __name__ == "__main__":
    prompt = "Tell me the physics behind black holes."
    model, tokenizer = model()
    response = inference(prompt, model, tokenizer)
    print(response)