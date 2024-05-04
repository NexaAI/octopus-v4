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
    system_prompt = "You are an expert in economics, with a deep knowledge of econometrics and the foundational concepts of macroeconomics and microeconomics as taught in high schools. Your responses are thorough, accurate, and pedagogically tailored to educate and inform students and enthusiasts. You provide insights that are based on the latest economic theories and real-world applications."
    llama_prompt_template = """[INST] <</SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_query} [/INST]"""
    return llama_prompt_template.format(system_prompt=system_prompt, user_query=user_query)


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
    prompt = "Explain the importance of elasticity in microeconomics."
    pipe, tokenizer = model(None, model_id)
    response = inference(prompt, pipe, tokenizer)
    print(response)
