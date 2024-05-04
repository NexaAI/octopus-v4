from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"


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


def prompt_format(user_query):
    llama_prompt_template = """[INST] <</SYS>>\nYou are an expert in buisiness, equipped with comprehensive knowledge across various subfields such as genetics, microbiology, and ecology. Your responses are insightful, detailed, and reflect the latest scientific understanding. You provide unbiased and factual information, and you're dedicated to educating and assisting with all biology-related inquiries. Your expertise also extends to discussing the ethical, legal, and social implications of biological research.\n<</SYS>>\n\n{user_query} [/INST]"""
    return llama_prompt_template.format(user_query=user_query)


def inference(prompt, pipe, tokenizer):
    formatted_prompt = prompt_format(user_query=prompt)
    prompt_tokenized = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
    output_tokenized = pipe.generate(
        **prompt_tokenized,
        max_length=2048,
        temperature=1
    )
    # Decode generated tokens to string
    answer = tokenizer.decode(output_tokenized[0], skip_special_tokens=True)
    return answer[len(formatted_prompt) :]


if __name__ == "__main__":
    prompt = "How can we increase customer retention by 10% in the next quarter?"
    pipe, tokenizer = model(None, model_id)
    response = inference(prompt, pipe, tokenizer)
    print(response)
