import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder


DIRECTORY_BASE = os.path.dirname(os.path.abspath(__file__))
full_path = os.path.join(DIRECTORY_BASE, "health_insurance_dataset2.csv")
db_insurance = pd.read_csv(full_path, encoding='latin1')

#print(db_insurance.head())

db_normalized = db_insurance[['age','sex','region','exercise_frequency','bmi','marital_status','smoker','chronic_condition','occupation_risk','income','diet_quality','education','alcohol_consumption','num_dependents_kids','num_dependents_adult','num_dependents_older','coverage_level','genetic_risk','use_last_year','charges']]
#print(db_normalized.head())
db_normalized.dropna(inplace=True)
#db_normalized = db_normalized[db_normalized["Health Score"] > 5]
#db_normalized = db_normalized[db_normalized["Health Score"] < 50]

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
    db_normalized["sex_e"] = encoder.fit_transform(db_normalized[["sex"]])
    db_normalized["region_e"] = encoder.fit_transform(db_normalized[["region"]])
    db_normalized["diet_quality_e"] = encoder.fit_transform(db_normalized[["diet_quality"]])
    db_normalized["exercise_frequency_e"] = encoder.fit_transform(db_normalized[["exercise_frequency"]])
    db_normalized["bmi_e"] = encoder.fit_transform(db_normalized[["bmi"]])
    db_normalized["marital_status_e"] = encoder.fit_transform(db_normalized[["marital_status"]])
    db_normalized["education_e"] = encoder.fit_transform(db_normalized[["education"]])
    db_normalized["alcohol_consumption_e"] = encoder.fit_transform(db_normalized[["alcohol_consumption"]])
    db_normalized["coverage_level_e"] = encoder.fit_transform(db_normalized[["coverage_level"]])
    db_normalized["genetic_risk_e"] = encoder.fit_transform(db_normalized[["genetic_risk"]])
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