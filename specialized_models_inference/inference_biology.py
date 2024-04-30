import transformers

# https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
model_id = "jondurbin/bagel-8b-v1.0"

def model(accelerator, model_name_or_id = model_id):
    # Setting up the model with necessary parameters
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name_or_id,
                                                              device_map={"": accelerator.process_index} if accelerator else "auto",
                                                              torch_dtype="auto", 
                                                              trust_remote_code=True
                                                              )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_id)
    return model, tokenizer

if __name__ == "__main__":
    model, tokenizer = model(None, model_id)