import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder


DIRECTORY_BASE = os.path.dirname(os.path.abspath(__file__))
full_path = os.path.join(DIRECTORY_BASE, "train.csv")
db_insurance = pd.read_csv(full_path, encoding='latin1')

#print(db_insurance.head())

db_normalized = db_insurance[["Age", "Gender", "Annual Income", "Number of Dependents", "Health Score", "Exercise Frequency", "Smoking Status", "Premium Amount"]]
#print(db_normalized.head())
db_normalized.dropna(inplace=True)
db_normalized = db_normalized[db_normalized["Health Score"] > 5]
db_normalized = db_normalized[db_normalized["Health Score"] < 50]

def show_age_income(db_normalized):

    xmin = db_normalized["Annual Income"].quantile(0.01)
    xmax = db_normalized["Annual Income"].quantile(0.95)

    sns.scatterplot(x="Age", y="Annual Income", data= db_normalized)
    plt.ylim(xmin, xmax)
    plt.grid(True)
    plt.show()

def show_age_exercice(db_normalized):
    xmin = db_normalized["Age"].quantile(0.01)
    xmax = db_normalized["Age"].quantile(0.95)

    sns.scatterplot(y="Age", x="Exercise Frequency", data= db_normalized)
    #plt.xlim(xmin, xmax)
    plt.grid(True)
    plt.show()

def show_health_exercice(db_normalized):
    xmin = db_normalized["Health Score"].quantile(0.01)
    xmax = db_normalized["Health Score"].quantile(0.95)

    sns.scatterplot(y="Health Score", x="Exercise Frequency", data= db_normalized)
    #plt.xlim(xmin, xmax)
    plt.grid(True)
    plt.show()

def show_gender_health(db_normalized):
    xmin = db_normalized["Health Score"].quantile(0.01)
    xmax = db_normalized["Health Score"].quantile(0.95)

    sns.scatterplot(y="Health Score", x="Gender", data= db_normalized)
    #plt.xlim(xmin, xmax)
    plt.grid(True)
    plt.show()

def show_gender_income(db_normalized):
    xmin = db_normalized["Annual Income"].quantile(0.01)
    xmax = db_normalized["Annual Income"].quantile(0.95)

    sns.scatterplot(y="Annual Income", x="Gender", data= db_normalized)
    #plt.xlim(xmin, xmax)
    plt.grid(True)
    plt.show()

def correlation_numeric(db_normalized):
    correlation(db_normalized.corr(numeric_only=True))

def correlation_spearman(db_normalized):
    correlation(db_normalized.corr(numeric_only=True, method="spearman"))

def correlation_numeric_encoded(db_normalized):
    encoder = OrdinalEncoder()
    db_normalized["Gender_Encoded"] = encoder.fit_transform(db_normalized[["Gender"]])
    db_normalized["Exercise_Frequency_Encoded"] = encoder.fit_transform(db_normalized[["Exercise Frequency"]])
    db_normalized["Smoking Status"] = encoder.fit_transform(db_normalized[["Smoking Status"]])
    #correlation_numeric(db_normalized)
    correlation_spearman(db_normalized)

def correlation(correlation_matrix):
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title("Matriz de Correlação")
    plt.show()

def pairplot(db_normalized, field):
    sns.pairplot(db_normalized, hue=field, corner=True)
    plt.show()

correlation_numeric_encoded(db_normalized)
#pairplot(db_normalized, "Gender")
#show_age_income(db_normalized)
#show_age_exercice(db_normalized)
#show_health_exercice(db_normalized)
#show_gender_health(db_normalized)
#show_gender_income(db_normalized)