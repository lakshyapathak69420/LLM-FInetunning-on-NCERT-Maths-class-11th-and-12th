
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from config import MODEL_ID, DATA_PATH, BATCH_SIZE

def load_and_prepare_data():
    """
    Loads the tokenized data, sets the tensor format, and initializes the tokenizer.
    
    Returns:
        tuple: (tokenised_dataset, tokenizer)
    """
    print(f"Loading data from {DATA_PATH}...")
    try:
        # Load the tokenized DataFrame from the pickle file
        tokenised_data = pd.read_pickle(DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_PATH}")
        return None, None
        
    # Convert the pandas DataFrame to a Hugging Face Dataset
    tokenised_df = Dataset.from_pandas(tokenised_data)

    # Set the format to PyTorch tensors, using only the necessary columns for the model
    tokenised_df.set_format(
        type="torch",
        columns=["attention_mask", "input_ids"]
    )

    # Initialize the tokenizer
    print(f"Loading tokenizer for {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # Add a padding token if the tokenizer doesn't have one (common for Causal LMs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return tokenised_df, tokenizer

def create_dataloader(tokenised_df, tokenizer):
    """
    Creates a DataLoader with padding for batching.
    
    Args:
        tokenised_df (datasets.Dataset): The prepared dataset.
        tokenizer (transformers.AutoTokenizer): The tokenizer for padding.
        
    Returns:
        torch.utils.data.DataLoader: The configured DataLoader.
    """
    # Data collator handles dynamic padding of sequences to the length of the longest in the batch
    data_collator = DataCollatorWithPadding(tokenizer, padding=True)
    
    # Create the DataLoader for iterating through the dataset in batches
    dataloader = DataLoader(
        tokenised_df,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=data_collator
    )
    print(f"DataLoader created with batch size {BATCH_SIZE}.")
    return dataloader