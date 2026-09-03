import os
import random
import numpy as np
import torch

from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
)

# Configurações do experimento
MODEL_ID = "allenai/OLMo-1B-hf"
DATA_FILE = "dolma_tokens/dolma_80m_final.txt"
OUTPUT_DIR = "./olmo_dolma_80m"
FINAL_DIR = "./olmo_dolma_80m"
BLOCK_SIZE = 512
SEED = 42

# Reprodutibilidade
set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    print(f"Treinando na GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Aviso: CUDA não disponível, rodando na CPU.")

# Tokenizer e Configuração
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, clean_up_tokenization_spaces=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

config = AutoConfig.from_pretrained(MODEL_ID)
config.hidden_size = 640
config.intermediate_size = 2560
config.num_hidden_layers = 16
config.num_attention_heads = 10
if hasattr(config, "num_key_value_heads"):
    config.num_key_value_heads = 10

# Inicialização do modelo do zero
model = AutoModelForCausalLM.from_config(config)
if hasattr(model.config, "use_cache"):
    model.config.use_cache = False

print(f"Modelo criado com {model.num_parameters():,} parâmetros.")

# Processamento do Dataset
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"Arquivo não encontrado: {DATA_FILE}")

dataset = load_dataset("text", data_files=DATA_FILE, split="train")

def tokenize_function(examples):
    return tokenizer(examples["text"], add_special_tokens=False)

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    batch_size=1000,
    remove_columns=["text"],
    desc="Tokenizando",
)

def group_texts(examples):
    concatenated = [tok for seq in examples["input_ids"] for tok in seq]
    total_length = (len(concatenated) // BLOCK_SIZE) * BLOCK_SIZE
    
    input_ids = [concatenated[i : i + BLOCK_SIZE] for i in range(0, total_length, BLOCK_SIZE)]
    attention_masks = [[1] * BLOCK_SIZE for _ in range(len(input_ids))]

    return {"input_ids": input_ids, "attention_mask": attention_masks}

lm_dataset = tokenized_dataset.map(
    group_texts,
    batched=True,
    batch_size=1000,
    desc="Criando blocos",
)

# Treinamento
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    fp16=torch.cuda.is_available(),
    optim="adamw_torch",
    learning_rate=3e-4,
    weight_decay=0.1,
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    report_to="none",
    dataloader_num_workers=0,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

print("Iniciando treinamento...")
train_result = trainer.train()

# Salvando artefatos
os.makedirs(FINAL_DIR, exist_ok=True)
trainer.save_model(FINAL_DIR)
tokenizer.save_pretrained(FINAL_DIR)

config_file = os.path.join(FINAL_DIR, "experiment_config.txt")
with open(config_file, "w", encoding="utf-8") as f:
    f.write(f"Model ID: {MODEL_ID}\n")
    f.write(f"Seed: {SEED}\n")
    f.write(f"Hidden size: {config.hidden_size}\n")
    f.write(f"Intermediate size: {config.intermediate_size}\n")
    f.write(f"Layers: {config.num_hidden_layers}\n")
    f.write(f"Attention heads: {config.num_attention_heads}\n")
    f.write(f"Block size: {BLOCK_SIZE}\n")
    f.write("Epochs: 1\nLearning rate: 3e-4\nWeight decay: 0.1\n")
    f.write("Batch size: 1\nGradient accumulation: 8\nInitialization: random\n")

print(f"Treinamento concluído. Modelo salvo em: {FINAL_DIR}")