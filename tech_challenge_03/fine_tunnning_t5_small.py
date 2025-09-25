# ====================================================
# 1. Instalação e imports
# ====================================================
# Antes de rodar no VS Code, instale no terminal:
# pip install transformers datasets accelerate
import pandas as pd
import os
from datasets import load_dataset
import torch 
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments
)

# ====================================================
# 2. Configuração do modelo
# ====================================================
MODEL_NAME = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Garantir pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ====================================================
# 3. Funções auxiliares
# ====================================================

def load_and_prepare_dataset(file_path, max_samples=50000):
    dataset = load_dataset("json", data_files=file_path)
    print(f"✅ Dataset carregado. Estrutura: {dataset}")

    split_name = list(dataset.keys())[0]
    available_columns = dataset[split_name].column_names
    print(f"📊 Colunas disponíveis: {available_columns}")

    if "title" not in available_columns or "content" not in available_columns:
        raise ValueError(f"Dataset deve conter colunas 'title' e 'content'. Encontradas: {available_columns}")

    dataset = dataset[split_name].select_columns(["title", "content"])

    if max_samples and len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))
        print(f"✂️ Dataset reduzido para {max_samples} amostras")

    def transform(example):
        return {
            "instruction": "Generate the product description from the title",
            "input": example["title"],
            "output": example["content"]
        }

    dataset = dataset.map(transform, remove_columns=["title", "content"])
    print(f"📦 Dataset final no formato instruction/input/output: Dataset({dataset})")
    return dataset


def preprocess_function(examples):
    inputs = []
    targets = []

    for instr, title, content in zip(examples["instruction"], examples["input"], examples["output"]):
        if title and content and isinstance(title, str) and isinstance(content, str):
            title = title.strip()
            content = content.strip()

            if len(title) > 2 and len(content) > 10:
                inputs.append(f"{instr}: {title}")
                targets.append(content)

    model_inputs = tokenizer(
        inputs,
        max_length=64,
        truncation=True,
        padding="max_length"
    )

    labels = tokenizer(
        targets,
        max_length=256,
        truncation=True,
        padding="max_length"
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# ====================================================
# 4. Carregar e tokenizar dataset
# ====================================================
file_path = "/Users/izabela.oliveira/Documents/GitHub/POS-IA/tech_challenge_03/trn.json.gz"  

dataset = load_and_prepare_dataset(file_path, max_samples=50000)

tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset.column_names
)

# Dividir em treino/validação
train_test = tokenized_dataset.train_test_split(test_size=0.1)
train_dataset = train_test["train"]
eval_dataset = train_test["test"]

# ====================================================
# 5. Configuração do treinamento
# ====================================================
batch_size = 8

args = TrainingArguments(
    output_dir="./results",
    eval_strategy="steps",  
    eval_steps=1000,
    save_strategy="steps",
    save_steps=1000,
    logging_steps=200,
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=4,  # simula batch maior
    num_train_epochs=3,             
    weight_decay=0.01,
    fp16=torch.cuda.is_available(), # só ativa se tiver GPU
    report_to="none"
)

# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# ====================================================
# 6. Criar trainer
# ====================================================
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# ====================================================
# 7. Avaliar e treinar 🚀
# ====================================================
results = trainer.evaluate(eval_dataset)
print("📊 Resultados da avaliação:", results)

trainer.train()

# Salvar modelo e tokenizer
save_dir = "/Users/izabela.oliveira/Documents/GitHub/POS-IA/tech_challenge_03/meu_modelo_finetuned"
trainer.save_model(save_dir)
tokenizer.save_pretrained(save_dir)
print("💾 Modelo e tokenizer salvos em:", save_dir)