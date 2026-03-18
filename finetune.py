import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq, AdamW
from tqdm import tqdm
from src.model_utils import get_model_and_tokenizer, apply_lora
from src.dataset_utils import load_math_dataset, format_to_chatml

# Config
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DATA_PATH = "data/math_qa_dataset.json"
SYSTEM_PROMPT = "You are a math expert who provides detailed, step-by-step solutions for NCERT problems"
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 3e-4

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = get_model_and_tokenizer(MODEL_NAME)
    model = apply_lora(model).to(device)

    # Prepare Data
    df = load_math_dataset(DATA_PATH)
    dataset = format_to_chatml(df, tokenizer, SYSTEM_PROMPT)
    
    # Tokenize
    tokenized_ds = dataset.map(
        lambda x: tokenizer(x["text"], truncation=True, max_length=128), 
        batched=False
    )
    
    # Create labels (masking system/user prompts would be better, but this follows notebook logic)
    def add_labels(example):
        example["labels"] = example["input_ids"].copy()
        return example
    
    tokenized_ds = tokenized_ds.map(add_labels)
    tokenized_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
    dataloader = DataLoader(tokenized_ds, batch_size=BATCH_SIZE, collate_fn=data_collator, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1} Average Loss: {total_loss/len(dataloader)}")

    model.save_pretrained("lora_adapters")
    print("Training complete. Adapters saved to 'lora_adapters'.")

if __name__ == "__main__":
    train()