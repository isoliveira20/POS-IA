#Importanto bibliotecas
import requests
import pandas as pd
import time
import csv

#importando dados
dataset = "IINOVAII/Insurance_Dataset"
config = "default"
split = "train"
batch_size = 100
total_rows = 1200000
api_url = "https://datasets-server.huggingface.co/rows"
output_file = "insurance_data_full.csv"

# Flag para escrever cabeçalho apenas uma vez
first_batch = True

for offset in range(0, total_rows, batch_size):
    params = {
        "dataset": dataset,
        "config": config,
        "split": split,
        "offset": offset,
        "length": batch_size
    }

    response = requests.get(api_url, params=params)
    
    if response.status_code != 200:
        print(f"Erro na requisição com offset={offset}: {response.status_code}")
        break

    batch = response.json().get("rows", [])
    if not batch:
        print(f"Nenhum dado retornado no offset={offset}. Encerrando.")
        break

    records = [row["row"] for row in batch]
    df = pd.DataFrame(records)

    # Salvar no CSV (append)
    df.to_csv(output_file, mode='a', index=False, header=first_batch)

    first_batch = False  # Após a primeira escrita, desativa o cabeçalho

    # Exibir progresso
    if offset % 10000 == 0:
        print(f"{offset} linhas processadas...")

    # Evitar sobrecarga no servidor
    time.sleep(0.2)

print("Download completo. Arquivo salvo como insurance_data_full.csv.")

#Carregamento  da base de dados
df = pd.read_csv('insurance_data_full.csv')

#print(df.head()) #imprime as 5 primeiras linhas

