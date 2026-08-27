from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

caminho_modelo = "olmo_dolma_10m" 

print("Carregando o modelo e o tokenizador...")
tokenizer = AutoTokenizer.from_pretrained(caminho_modelo, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(caminho_modelo, trust_remote_code=True)

prompt = "Artificial intelligence is a field that"
inputs = tokenizer(prompt, return_tensors="pt")

print("Gerando texto...")
outputs = model.generate(
    **inputs, 
    max_new_tokens=50, 
    do_sample=True, 
    temperature=0.7
)

resultado = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n--- Resultado ---")
print(resultado)