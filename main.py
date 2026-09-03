from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

caminho_modelo = "olmo_dolma_80m" 

print("Carregando o modelo e o tokenizador...")
tokenizer = AutoTokenizer.from_pretrained(caminho_modelo, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(caminho_modelo, trust_remote_code=True)

prompt = "Question: What is the capital of France?\nAnswer:"
inputs = tokenizer(prompt, return_tensors="pt")

print("Gerando texto...")
outputs = model.generate(
    **inputs,
    max_new_tokens=5,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id,
)

resultado = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n--- Resultado ---")
print(resultado)