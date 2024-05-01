from transformers import AutoModelForCausalLM, AutoTokenizer

# https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
model_id = "jondurbin/bagel-8b-v1.0"


def model(accelerator, model_name_or_id=model_id):
    # Setting up the model with necessary parameters
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_id,
        device_map={"": accelerator.process_index} if accelerator else "auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_id, trust_remote_code=True)
    return model, tokenizer


def prepare_chat(user_query, tokenizer):
    # Define a chat template
    chat_template = [
        {"role": "system", "content": "You are biology expert, equipped with comprehensive knowledge across various subfields such as genetics, microbiology, and ecology."},
        {"role": "user", "content": user_query}
    ]
    # Prepare the chat for the model (tokenize the chat if necessary)
    prepared_chat = tokenizer.apply_chat_template(chat_template, tokenize=False)
    return prepared_chat


def inference(prompt, pipe, tokenizer):
    prompt = prepare_chat(prompt, tokenizer)
    prompt_tokenized = tokenizer(prompt, return_tensors="pt").to("cuda")
    # Generate output from the model
    output_tokenized = pipe.generate(
        **prompt_tokenized,
        max_length=2048,
        temperature=1
    )
    answer = tokenizer.decode(output_tokenized[0], skip_special_tokens=True)
    return answer


if __name__ == "__main__":
    prompt = "What are the primary functions of ribosomes in a cell?"
    pipe, tokenizer = model(None, model_id)
    response = inference(prompt, pipe, tokenizer)
    print(response)
