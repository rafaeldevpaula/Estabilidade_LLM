import os
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN não encontrado no .env")

REPO_ID = "allenai/dolma"
VERSION = "v1_6-sample"
TARGET_TOKENS = 10_000_000
OUTPUT_FILE = "dolma_10m.txt"
TOKENIZER_ID = "allenai/OLMo-1B-hf"

print("Carregando tokenizador...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, token=HF_TOKEN)

url_list_file = hf_hub_download(
    repo_id=REPO_ID,
    filename=f"urls/{VERSION}.txt",
    repo_type="dataset",
    token=HF_TOKEN,
)

with open(url_list_file, "r", encoding="utf-8") as f:
    urls = [line.strip() for line in f if line.strip()]

print(f"Iniciando download de {TARGET_TOKENS:,} tokens a partir de {len(urls)} arquivos...")

current_tokens = 0
documents = 0

with open(OUTPUT_FILE, "w", encoding="utf-8") as output:
    for idx, url in enumerate(urls, start=1):
        if current_tokens >= TARGET_TOKENS:
            break

        try:
            dataset = load_dataset("json", data_files=url, split="train", streaming=True)
        except Exception as e:
            print(f"Erro ao acessar {url}: {e}. Pulando...")
            continue

        for example in dataset:
            if current_tokens >= TARGET_TOKENS:
                break

            text = example.get("text", "")
            if not text:
                continue

            token_ids = tokenizer.encode(text, add_special_tokens=False)
            num_tokens = len(token_ids)
            remaining = TARGET_TOKENS - current_tokens

            if num_tokens <= remaining:
                output.write(text.replace("\n", " ") + "\n")
                current_tokens += num_tokens
            else:
                final_text = tokenizer.decode(token_ids[:remaining], skip_special_tokens=True)
                output.write(final_text.replace("\n", " ") + "\n")
                current_tokens += remaining

            documents += 1

            if documents % 1000 == 0:
                print(f"Progresso: {current_tokens:,}/{TARGET_TOKENS:,} tokens ({(current_tokens/TARGET_TOKENS)*100:.1f}%)")

print(f"\nConcluído! {current_tokens:,} tokens salvos em '{OUTPUT_FILE}' ({documents:,} documentos).")