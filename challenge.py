#Importanto bibliotecas
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score 

#Exploração de dados:
pd.set_option('display.max_columns', None)

#Carregamento da base de dados e exploração das suas características
df = pd.read_csv('insurance_dataset.csv')
print(df.head()) #imprime as 5 primeiras linhas
print(df.shape)  # Verifica quantas linhas e colunas tem


print(df.info())       # Tipos de dados e colunas nulas
print(df.describe())   # Estatísticas para colunas numéricas
print(df.columns)      # Lista os nomes das colunas

#Valores nulos
print(df.isnull().sum()) # imprime a soma de valores nulos por coluna

#Valores duplicados
print(df.duplicated().sum()) #imprime a soma de valores duplicados por coluna

#Visualizar a distribuição das variáveis
#sns.pairplot(df)
#plt.show()


# Tratamento de variáveis categóricas p/ numéricas usando LabelEncoder
label_encoder = LabelEncoder()
for col in ['Gender', 'Marital Status', 'Education Level', 'Occupation', 'Location', 'Policy Type', 'Policy Start Date', 'Customer Feedback', 'Smoking Status', 'Exercise Frequency', 'Property Type']:
    df[col] = label_encoder.fit_transform(df[col])

print(df.head()) # Verificando a transformação

# Dividir os dados em variáveis features (x) e target (y)
x = df.drop('Premium Amount', axis=1)  # Dados de entrada - todas as colunas menos 'charges'
y = df['Premium Amount']  # A coluna 'charges' é o target, variável que queremos prever

# Dividindo o conjunto de dados em treino e teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test= train_test_split(x, y, test_size=0.2, random_state=42)

# Dados de treino - Normalizar os dados numéricos (colocar tudo na mesma escala — geralmente de 0 a 1)
scaler = MinMaxScaler() # Inicializa o MinMaxScaler
features = df.drop('Premium Amount', axis=1).columns # Nome das colunas dos dados de entrada
X_train_scaled = scaler.fit_transform(X_train)  # aprende (fit) e aplica (transform) no treino
X_test_scaled = scaler.transform(X_test)        # aprende (fit) e aplica (transform) a mesma escala no teste
df_scaled = scaler.fit_transform(df[features]) # Cria um novo DataFrame com os dados normalizados


# Visualizando a matriz de correlação
plt.figure(figsize=(20, 12))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Matriz de correlação')
plt.show()