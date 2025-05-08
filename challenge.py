#Importanto bibliotecas
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score 

#Exploração de dados:
pd.set_option('display.max_columns', None)

#Carregamento da base de dados e exploração das suas características
df = pd.read_csv('insurance_data_full.csv')
#print(df.head()) #imprime as 5 primeiras linhas

print(df.describe()) #imprime estatísticas descritivas


print(df.info()) 