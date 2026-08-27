from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "allenai/OLMo-1B-hf"
)

total_tokens = 0

with open("dolma_10m.txt", "r", encoding="utf-8") as f:
    for line in f:
        total_tokens += len(
            tokenizer.encode(
                line,
                add_special_tokens=False
            )
        )

print(f"Total de tokens: {total_tokens:,}")