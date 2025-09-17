import pandas as pd
from bs4 import BeautifulSoup as bs
import requests
import regex as re
import openai
from datasets import load_dataset
from datasets import Dataset
from trl import SFTTrainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score


# Carregamento da base de dados
df = pd.read_json('/Users/izabela.oliveira/Documents/GitHub/Pos-tech/Tech_Challenge_03/LF-Amazon-1.3M/trn.json.gz', lines=True)
#print(df.head())

# Check da estrutura do DataFrame
#print(df.columns) #Index(['uid', 'title', 'content', 'target_ind', 'target_rel'], dtype='object')

# Seleção das colunas title e content
df = df[['title', 'content']]
#print(df.head())

# Check de valores nulos
#Soma de valores nulos por coluna
#print(df.isnull().sum())


#Verifica se existem valores duplicados
#print(df.duplicated().sum()) # imprime a soma de valores duplicados
df = df.drop_duplicates() # Remove valores duplicados
#print(df.duplicated().sum()) # Verifica se ainda existem valores duplicados

#Preparação dos dados
dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.2)

#Tokenização do dataset
model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch['content'], padding='max_length', truncation=True, max_length=512)
dataset = dataset.map(tokenize, batched=True, batch_size=len(dataset))

#Carregando modelo pré-treinado
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.resize_token_embeddings(len(tokenizer))

#Preparação do target
LabelEncoder = LabelEncoder()
df['label'] = LabelEncoder.fit_transform(df['title'])

dataset = Dataset.from_pandas(df[['content', 'label']])
dataset = dataset.train_test_split(test_size=0.2)

#Tokenização do dataset
def tokenize(batch):
    return tokenizer(batch['content'], padding='max_length', truncation=True, max_length=512)
dataset = dataset.map(tokenize, batched=True)

#Carregamento do modelo pré-treinado
num_labels = len(df['label'].unique())
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

#Configuração de treinamento
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    evaluation_strategy="steps",   # substitui evaluate_during_training
    eval_steps=500,                # avalia a cada 500 steps
    save_strategy="steps",         # também pode salvar checkpoints a cada X steps
    save_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model='accuracy'
)

# Função de avaliação
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return {'accuracy': accuracy, 'f1': f1}

# Configuração do treinador
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    compute_metrics=compute_metrics
)
# Início do treinamento
trainer.train()

#Salvando o modelo treinado
model.save_pretrained('./bert-finetuned')
#Salvando o tokenizador
tokenizer.save_pretrained('./bert-finetuned')
