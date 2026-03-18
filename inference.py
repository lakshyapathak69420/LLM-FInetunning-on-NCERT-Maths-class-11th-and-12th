import torch
from peft import PeftModel
from src.model_utils import get_model_and_tokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "lora_adapters"

def run_inference(prompt):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model, tokenizer = get_model_and_tokenizer(MODEL_NAME)
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH).to(device)
    
    messages = [
        {"role": "system", "content": "You are a math expert."},
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=256)
    
    # Remove input tokens from output
    response_ids = output_ids[0][len(inputs["input_ids"][0]):]
    print(tokenizer.decode(response_ids, skip_special_tokens=True))

if __name__ == "__main__":
    user_q = input("Enter a math question: ")
    run_inference(user_q)