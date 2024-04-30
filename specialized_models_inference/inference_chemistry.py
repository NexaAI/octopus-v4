from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

model_id = "AI4Chem/ChemLLM-7B-Chat"

def model(accelerator, model_name_or_id = model_id):
    pipe = AutoModelForCausalLM.from_pretrained(model_name_or_id,
                                                              device_map={"": accelerator.process_index} if accelerator else "auto",
                                                              torch_dtype="auto", 
                                                              trust_remote_code=True
                                                              )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_id, trust_remote_code=True)
    return pipe, tokenizer

def InternLM2_format(instruction, prompt):
    prefix_template = ["<|system|>:", "{}"]
    prompt_template = ["<|user|>:", "{}\n", "<|Bot|>:\n"]
    system = f"{prefix_template[0]}\n{prefix_template[-1].format(instruction)}\n"
    prompt = f"\n{prompt_template[0]}\n{prompt_template[1].format(prompt)}{prompt_template[-1]}"
    return f"{system}{prompt}"

def inference(prompt, pipe, tokenizer):
    formatted_prompt = InternLM2_format("You are a chemistry expert", prompt)
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
    generation_config = GenerationConfig(
        do_sample=True,
        top_k=1,
        temperature=0.9,
        max_new_tokens=500,
        repetition_penalty=1.5,
        pad_token_id=tokenizer.eos_token_id,
    )
    outputs = pipe.generate(**inputs, **generation_config.__dict__)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result[len(formatted_prompt):]


if __name__ == "__main__":
    prompt = "What is Molecule of Ibuprofen?"
    pipe, tokenizer = model(None)
    response = inference(prompt, pipe, tokenizer)
    print(response)