from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import pandas as p

save_dir = "/Users/izabela.oliveira/Documents/GitHub/POS-IA/tech_challenge_03/meu_modelo_finetuned"

tokenizer = AutoTokenizer.from_pretrained(save_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(save_dir)

print("✅ Modelo e tokenizer carregados com sucesso!")


#===========================================

# Criar pipeline de inferência
generator = pipeline("text2text-generation", model=save_dir, tokenizer=save_dir)

# Teste
title = "Tênis Nike Revolution 6 Masculino"
res = generator(f"Generate the product description from the title: {title}",
                max_new_tokens=100,
                do_sample=True,   # ativa sampling (mais criativo)
                top_k=50,
                top_p=0.95)

print("📦 Título:", title)
print("📝 Descrição gerada:", res[0]["generated_text"])

