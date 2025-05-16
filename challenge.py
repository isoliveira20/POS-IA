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
from sklearn.impute import SimpleImputer


#Exploração de dados:
pd.set_option('display.max_columns', None)

#Carregamento da base de dados e exploração das suas características
df = pd.read_csv('health_insurance_dataset.csv')
print(df.head()) #imprime as 5 primeiras linhas
print(df.shape)  # Verifica quantas linhas e colunas tem


print(df.info())       # Tipos de dados e colunas nulas
print(df.describe())   # Estatísticas para colunas numéricas
print(df.columns)      # Lista os nomes das colunas

#Valores nulos
print(df.isnull().sum()) # imprime a soma de valores nulos por coluna

#tratamento de valores nulos
df['bmi'] = df['bmi'].fillna(df['bmi'].mode())
df['alcohol_consumption'] = df['alcohol_consumption'].fillna(df['alcohol_consumption'].mode())
df['diet_quality'] = df['diet_quality'].fillna(df['diet_quality'].mode()[0])

#Verifica se ainda existem valores nulos
print(df.isnull().sum()) # imprime a soma de valores nulos por coluna

#Valores duplicados
print(df.duplicated().sum()) #imprime a soma de valores duplicados por coluna

#Visualizar a distribuição das variáveis
#sns.pairplot(df)
#plt.show()

# Tratamento de variáveis categóricas p/ numéricas usando LabelEncoder
label_encoder = LabelEncoder()
for col in ['sex','region','exercise_frequency','bmi','marital_status', 'alcohol_consumption','diet_quality','occupation_risk','coverage_level','genetic_risk','education']:
    df[col] = label_encoder.fit_transform(df[col])

print(df.head()) # Verificando a transformação

# Matriz de correlação
plt.figure(figsize=(12, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlação')
plt.show()

# Dividir os dados em variáveis features (x) e target (y)
x = df.drop('charges', axis=1)  # Dados de entrada - todas as colunas menos 'charges'
y = df['charges']  # A coluna 'Premium Amount' é o target, variável que queremos prever

# Dividindo o conjunto de dados em treino e teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test= train_test_split(x, y, test_size=0.2, random_state=42)

# == Regressão Linear == #
model = LinearRegression()
model.fit(X_train, y_train)  # Treinando o modelo com os dados de treino

# Fazendo previsões com os dados de teste
y_pred = model.predict(X_test)

# Avaliação do modelo
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2:", r2)

# Visualização dos resultados
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.xlabel('Valores Reais')
plt.ylabel('Valores Preditos')
plt.title('Valores Reais vs Valores Preditos')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.show()

# == KNN Regressor == #
from sklearn.neighbors import KNeighborsRegressor

# Criando o modelo KNN Regressor
knn_model = KNeighborsRegressor(n_neighbors=5)

# Treinando o modelo com os dados de treino
knn_model.fit(X_train, y_train)

# Fazendo previsões com os dados de teste
y_pred_knn = knn_model.predict(X_test)

# Avaliação do modelo
mae_knn = mean_absolute_error(y_test, y_pred_knn)
mse_knn = mean_squared_error(y_test, y_pred_knn)
rmse_knn = np.sqrt(mse_knn)
r2_knn = r2_score(y_test, y_pred_knn)
print("MAE KNN:", mae_knn)
print("MSE KNN:", mse_knn)
print("RMSE KNN:", rmse_knn)
print("R2 KNN:", r2_knn)

# Visualização dos resultados
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_knn)
plt.xlabel('Valores Reais')
plt.ylabel('Valores Preditos')
plt.title('Valores Reais vs Valores Preditos KNN')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.show()


# == Decision Tree Regressor == #
from sklearn.tree import DecisionTreeRegressor

# Criando o modelo Decision Tree Regressor
dt_model = DecisionTreeRegressor(random_state=42)

# Treinando o modelo com os dados de treino
dt_model.fit(X_train, y_train)

# Fazendo previsões com os dados de teste
y_pred_dt = dt_model.predict(X_test)

# Avaliação do modelo
mae_dt = mean_absolute_error(y_test, y_pred_dt)
mse_dt = mean_squared_error(y_test, y_pred_dt)
rmse_dt = np.sqrt(mse_dt)
r2_dt = r2_score(y_test, y_pred_dt)
print("MAE Decision Tree:", mae_dt)
print("MSE Decision Tree:", mse_dt)
print("RMSE Decision Tree:", rmse_dt)
print("R2 Decision Tree:", r2_dt)

# Visualização dos resultados
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_dt)
plt.xlabel('Valores Reais')
plt.ylabel('Valores Preditos')
plt.title('Valores Reais vs Valores Preditos Decision Tree')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.show()

# == Random Forest Regressor == #
from sklearn.ensemble import RandomForestRegressor
# Criando o modelo Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
# Treinando o modelo com os dados de treino
rf_model.fit(X_train, y_train)
# Fazendo previsões com os dados de teste
y_pred_rf = rf_model.predict(X_test)
# Avaliação do modelo
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print("MAE Random Forest:", mae_rf)
print("MSE Random Forest:", mse_rf)
print("RMSE Random Forest:", rmse_rf)
print("R2 Random Forest:", r2_rf)
# Visualização dos resultados
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_rf)
plt.xlabel('Valores Reais')
plt.ylabel('Valores Preditos')
plt.title('Valores Reais vs Valores Preditos Random Forest')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.show()

# == Lass Regression == #
from sklearn.linear_model import Lasso
# Criando o modelo Lasso Regression
lasso_model = Lasso(alpha=0.1)
# Treinando o modelo com os dados de treino
lasso_model.fit(X_train, y_train)
# Fazendo previsões com os dados de teste
y_pred_lasso = lasso_model.predict(X_test)
# Avaliação do modelo
mae_lasso = mean_absolute_error(y_test, y_pred_lasso)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
rmse_lasso = np.sqrt(mse_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)
print("MAE Lasso:", mae_lasso)
print("MSE Lasso:", mse_lasso)
print("RMSE Lasso:", rmse_lasso)
print("R2 Lasso:", r2_lasso)

#xgboost
from xgboost import XGBRegressor
# Criando o modelo XGBoost Regressor
xgb_model = XGBRegressor(n_estimators=100, random_state=42)
# Treinando o modelo com os dados de treino
xgb_model.fit(X_train, y_train)
# Fazendo previsões com os dados de teste
y_pred_xgb = xgb_model.predict(X_test)
# Avaliação do modelo
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mse_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)
print("MAE XGBoost:", mae_xgb)
print("MSE XGBoost:", mse_xgb)
print("RMSE XGBoost:", rmse_xgb)
print("R2 XGBoost:", r2_xgb)

