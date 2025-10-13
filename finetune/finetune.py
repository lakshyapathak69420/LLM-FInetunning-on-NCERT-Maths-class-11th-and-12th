from transformers import AutoModelForCausalLM
from peft import LoraConfig, LoraModel
import torch
from torch.optim import AdamW
from tqdm import tqdm

from config import (
    MODEL_ID, RANK, LORA_ALPHA, TARGET_MODULES, LORA_DROPOUT, TASK_TYPE,
    EPOCHS, LEARNING_RATE, DEVICE
)
from data_utils import load_and_prepare_data, create_dataloader


def setup_model_and_lora(tokenizer):
    """
    Loads the base model and configures it for LoRA finetuning.
    
    Args:
        tokenizer (transformers.AutoTokenizer): The tokenizer object.
        
    Returns:
        tuple: (lora_model, base_model)
    """
    print(f"Loading base model: {MODEL_ID}...")
 
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    

    model.config.pad_token_id = tokenizer.pad_token_id

    # Define the LoRA configuration
    lora_config = LoraConfig(
        r=RANK, 
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        task_type=TASK_TYPE
    )

    lora_model = LoraModel(model, lora_config, "default")
    
  
    lora_model.to(DEVICE)
    print("LoRA Model setup complete. Only LoRA parameters require gradients.")
    
    return lora_model, model


def get_trainable_parameters(lora_model):
    """
    Extracts only the trainable LoRA parameters for the optimizer.
    
    Args:
        lora_model (peft.LoraModel): The LoRA wrapped model.
        
    Returns:
        list: List of trainable torch parameters.
    """
    params = []
    
    for param in lora_model.parameters():
        if param.requires_grad:
            params.append(param)
    print(f"Found {len(params)} groups of trainable parameters.")
    return params


def train_lora_model(lora_model, dataloader):
    """
    Executes the main training loop.
    
    Args:
        lora_model (peft.LoraModel): The model to be trained.
        dataloader (torch.utils.data.DataLoader): The data iterator.
    """
    # Get only the trainable LoRA parameters
    trainable_params = get_trainable_parameters(lora_model)
    
    # Initialize the AdamW optimizer
    optimizer = AdamW(params=trainable_params, lr=LEARNING_RATE)
    
    # Set the model to training mode (enables dropout, etc.)
    lora_model.train()
    
    print(f"Starting training for {EPOCHS} epochs on {DEVICE}...")

    # List to store batch losses (optional, for plotting later)
    training_loss = [] 

    # Main training loop
    for epoch in range(EPOCHS):
        total_loss = 0
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
    
            batch = {name: tensor.to(DEVICE) for name, tensor in batch.items()}
            
            
            labels = batch["input_ids"] 

            outputs = lora_model(**batch, labels=labels)
            loss = outputs.loss
            
          
            loss.backward()
            
            
            optimizer.step()
            

            optimizer.zero_grad()
            
            
            total_loss += loss.item() 
            training_loss.append(loss.item())

        
        avg_loss = total_loss / len(dataloader)
        print(f"\nAverage loss for Epoch {epoch + 1} is {avg_loss:.4f}")

    print("Training finished!")
    


if __name__ == "__main__":
    tokenised_df, tokenizer = load_and_prepare_data()
    if tokenised_df is None:
        exit()
        
    dataloader = create_dataloader(tokenised_df, tokenizer)
    lora_model, base_model = setup_model_and_lora(tokenizer)

    train_lora_model(lora_model, dataloader)