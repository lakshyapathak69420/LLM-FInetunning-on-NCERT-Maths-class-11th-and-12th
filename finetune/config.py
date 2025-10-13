
import torch

# --- Model and LoRA Hyperparameters ---

MODEL_ID = "google/gemma-2b"

RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
TARGET_MODULES = ["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]
TASK_TYPE = "CAUSAL_LM"



EPOCHS = 3
LEARNING_RATE = 0.001
BATCH_SIZE = 4
DATA_PATH = "tokenised_prompt.pkl"

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")