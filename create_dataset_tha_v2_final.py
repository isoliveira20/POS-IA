import pandas as pd
import numpy as np
from scipy.stats import skewnorm, truncnorm
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================
# CONFIGURAÇÕES INICIAIS
# ==============================================
np.random.seed(42)
N_SAMPLES = 100_000
MISSING_PERCENT = 0.05
PLOT=False

# ==============================================
# FEATURES MAPEADAS
# ==============================================
AGE = 'age'
SEX = 'sex'
REGION = 'region'
BMI = 'bmi'
SMOKER = 'smoker'
CHRONIC_CONDITION = 'chronic_condition'
GENETIC_RISK = 'genetic_risk'
EXERCISE_FREQUENCY = 'exercise_frequency'
ALCOHOL_CONSUMPTION = 'alcohol_consumption'
DIET_QUALITY = 'diet_quality'
INCOME = 'income'
OCCUPATION_RISK = 'occupation_risk'
COVERAGE_LEVEL = 'coverage_level'
NUM_DEPENDENTS_KIDS = 'num_dependents_kids'
NUM_DEPENDENTS_ADULT = 'num_dependents_adult'
NUM_DEPENDENTS_OLDER = 'num_dependents_older'
EDUCATION = 'education'
MARITAL_STATUS = 'marital_status'
CHARGES = 'charges'
SALUBRITY = 'salubrity'
USE_LAST_YEAR = 'use_last_year'

# ==============================================
# GERAR VARIÁVEIS BASE
# ==============================================

def generate_age(n_samples):
    from scipy.stats import truncnorm
    lower, upper = 15, 90
    mu, sigma = 35, 20
    a, b = (lower - mu) / sigma, (upper - mu) / sigma
    return truncnorm.rvs(a, b, loc=mu, scale=sigma, size=n_samples).astype(int)


def generate_sex(age_array):
    sexes = []
    for age in age_array:
        if age <= 24:
            sex = np.random.choice(['male', 'female'], p=[0.51, 0.49])
        else:
            sex = np.random.choice(['male', 'female'], p=[0.47, 0.53])
        sexes.append(sex)
    return np.array(sexes)


def generate_marital_status(age_array):
    result = []
    for age in age_array:
        if age < 25:
            result.append(np.random.choice(['single', 'married', 'divorced', 'widowed'], p=[0.85, 0.10, 0.04, 0.01]))
        elif age < 35:
            result.append(np.random.choice(['single', 'married', 'divorced', 'widowed'], p=[0.40, 0.50, 0.08, 0.02]))
        elif age < 50:
            result.append(np.random.choice(['single', 'married', 'divorced', 'widowed'], p=[0.20, 0.65, 0.12, 0.03]))
        elif age < 65:
            result.append(np.random.choice(['single', 'married', 'divorced', 'widowed'], p=[0.15, 0.60, 0.15, 0.10]))
        else:
            result.append(np.random.choice(['single', 'married', 'divorced', 'widowed'], p=[0.10, 0.40, 0.10, 0.40]))
    return np.array(result)

def generate_education(age_array, income_array):
    result = []
    for age, income in zip(age_array, income_array):
        if age < 20:
            result.append(np.random.choice(['primary', 'high_school'], p=[0.2, 0.8]))
        elif age < 25:
            result.append(np.random.choice(['high_school', 'college'], p=[0.3, 0.7]))
        elif age >= 25 and income > 100000:
            result.append(np.random.choice(['high_school', 'college', 'postgraduate'], p=[0.05, 0.5, 0.45]))
        elif age >= 25 and income > 40000:
            result.append(np.random.choice(['high_school', 'college'], p=[0.4, 0.6]))
        elif age >= 25:
            result.append(np.random.choice(['primary', 'high_school'], p=[0.3, 0.7]))
        else:
            result.append('high_school')
    return np.array(result)

def generate_education_v2(data):
    n = len(data)
    education = []
    for i, (age,income) in enumerate(zip(df[AGE], df[INCOME])):
        if age < 20:
            education.append(np.random.choice(['primary', 'high_school'], p=[0.2, 0.8]))
        elif age < 25:
            education.append(np.random.choice(['high_school', 'college'], p=[0.3, 0.7]))
        elif age >= 25 and income > 100000:
            education.append(np.random.choice(['high_school', 'college', 'postgraduate'], p=[0.05, 0.5, 0.45]))
        elif age >= 25 and income > 40000:
            education.append(np.random.choice(['high_school', 'college'], p=[0.4, 0.6]))
        elif age >= 25:
            education.append(np.random.choice(['primary', 'high_school'], p=[0.3, 0.7]))
        else:
            education.append('high_school')
    
    return pd.Series(education, index=data.index)

def generate_exercise_frequency(age_array, sex_array):
    result = []
    for age, sex in zip(age_array, sex_array):
        if sex == 'male':
            if age < 25:
                p = [0.2, 0.3, 0.3, 0.2]
            elif age < 40:
                p = [0.25, 0.35, 0.25, 0.15]
            elif age < 60:
                p = [0.3, 0.4, 0.2, 0.1]
            else:
                p = [0.4, 0.4, 0.15, 0.05]
        else:
            if age < 25:
                p = [0.25, 0.35, 0.25, 0.15]
            elif age < 40:
                p = [0.3, 0.4, 0.2, 0.1]
            elif age < 60:
                p = [0.35, 0.4, 0.2, 0.05]
            else:
                p = [0.45, 0.4, 0.1, 0.05]
        result.append(np.random.choice(['sedentary', 'light', 'moderate', 'active'], p=p))
    return np.array(result)


def generate_bmi(age_array, region_array, exercise_array):
    result = []
    for age, region, exercise_frequency in zip(age_array, region_array, exercise_array):
        if age < 26:
            loc, scale, a = 23, 3, 2
        elif age < 55:
            loc, scale, a = 27, 4, 5
        elif age < 65:
            loc, scale, a = 28, 4.5, 6
        else:
            loc, scale, a = 27, 5, 3

        bmi_factor_by_region = {
            'north': -0.5,
            'northeast': -0.3,
            'southeast': 0.0,
            'south': 0.3,
            'central_west': -0.2
        }
        bmi_adjustment_by_exercise = {
            'sedentary': 3.0,
            'light': 1.5,
            'moderate': -1.5,
            'active': -3.0
        }
        base_loc = loc + bmi_factor_by_region.get(region, 0) + bmi_adjustment_by_exercise.get(exercise_frequency, 0)
        bmi_raw = skewnorm.rvs(a, loc=base_loc, scale=scale)
        result.append(np.clip(bmi_raw, 16, 45))
    return np.array(result)


def generate_smoker(age_array, sex_array):
    result = []
    for age, sex in zip(age_array, sex_array):
        if sex == 'male':
            if age < 25:
                p = 0.15
            elif age < 40:
                p = 0.40
            elif age < 60:
                p = 0.35
            else:
                p = 0.20
        else:
            if age < 25:
                p = 0.12
            elif age < 40:
                p = 0.15
            elif age < 60:
                p = 0.18
            else:
                p = 0.10
        result.append(np.random.binomial(1, p))
    return np.array(result)


def generate_chronic_condition_v2(age_array):
    result = []
    for age in age_array:
        if age < 30:
            p = [0.90, 0.08, 0.02]  # 90% sem doença, 8% leve, 2% moderada+
        elif age < 50:
            p = [0.75, 0.2, 0.05]
        elif age < 65:
            p = [0.55, 0.3, 0.15]
        else:
            p = [0.35, 0.4, 0.25]
        result.append(np.random.choice([0, 1, 2], p=p))
    return np.array(result)

def generate_alcohol_consumption(age_array, sex_array, income_array, region_array):
    result = []
    for age, sex, income, region in zip(age_array, sex_array, income_array, region_array):
        base = 1
        if sex == 'male':
            base += 1
        if age > 50:
            base -= 0.5
        if income > 80000:
            base += 0.5
        if region in ['south', 'southeast']:
            base += 0.5
        # Consumo em doses por semana, com variação aleatória
        consumption = np.random.poisson(max(base, 0))
        result.append(consumption)
    return np.array(result)

def generate_alcohol_consumption_v2(data):
    n = len(data)
    consumption = np.zeros(n)
    
    for i, (age, sex, income, region) in enumerate(zip(df[AGE], df[SEX], df[INCOME], df[REGION])):
        base = 1
        if sex == 'male':
            base += 1
        if age > 50:
            base -= 0.5
        if income > 80000:
            base += 0.5
        if region in ['south', 'southeast']:
            base += 0.5
        # Consumo em doses por semana, com variação aleatória
        consumption[i] = np.random.poisson(max(base, 0))
        
    return pd.Series(consumption, index=data.index)


def generate_diet_quality(income_array):
    result = []
    for income in income_array:
        if income < 30000:
            p = [0.6, 0.3, 0.1]  # poor, average, good
        elif income < 80000:
            p = [0.3, 0.5, 0.2]
        else:
            p = [0.1, 0.4, 0.5]
        result.append(np.random.choice(['poor', 'average', 'good'], p=p))
    return np.array(result)


def generate_occupation_risk(age_array, region_array, income_array, sex_array):
    result = []
    for age, region, income, sex in zip(age_array, region_array, income_array, sex_array):
        base_probs = np.array([0.4, 0.3, 0.15, 0.10, 0.05])

        age_adj = np.array([-0.05, -0.05, 0.05, 0.03, 0.02]) if age < 30 else np.array([0.1, 0.05, -0.05, -0.05, -0.05])
        region_adj_map = {
            'north': np.array([0.05, 0.0, -0.02, -0.02, -0.01]),
            'northeast': np.array([0.03, 0.0, -0.01, -0.01, -0.01]),
            'southeast': np.array([-0.02, 0.0, 0.01, 0.01, 0.0]),
            'south': np.array([-0.03, -0.02, 0.02, 0.02, 0.01]),
            'central_west': np.array([-0.01, 0.0, 0.0, 0.0, 0.01])
        }
        region_adj = region_adj_map.get(region, np.zeros(5))
        income_adj = np.array([-0.05, -0.05, 0.05, 0.03, 0.02]) if income < 30000 else (np.array([0.05, 0.03, -0.03, -0.03, -0.02]) if income > 80000 else np.zeros(5))
        sex_adj = np.array([-0.05, -0.03, 0.03, 0.03, 0.02]) if sex == 'male' else np.array([0.05, 0.03, -0.03, -0.03, -0.02])

        probs = base_probs + age_adj + region_adj + income_adj + sex_adj
        probs = np.clip(probs, 0.001, None)
        probs /= probs.sum()
        choice = np.random.choice([1, 2, 3, 4, 5], p=probs)
        result.append(choice)

    return np.array(result)


def generate_occupation_risk_v2(data):
    result = []
    for i, (age, region, sex) in enumerate(zip(data[AGE], data[REGION], data[SEX])):
        base_probs = np.array([0.4, 0.3, 0.15, 0.10, 0.05])

        age_adj = np.array([-0.05, -0.05, 0.05, 0.03, 0.02]) if age < 30 else np.array([0.1, 0.05, -0.05, -0.05, -0.05])
        region_adj_map = {
            'north': np.array([0.05, 0.0, -0.02, -0.02, -0.01]),
            'northeast': np.array([0.03, 0.0, -0.01, -0.01, -0.01]),
            'southeast': np.array([-0.02, 0.0, 0.01, 0.01, 0.0]),
            'south': np.array([-0.03, -0.02, 0.02, 0.02, 0.01]),
            'central_west': np.array([-0.01, 0.0, 0.0, 0.0, 0.01])
        }
        region_adj = region_adj_map.get(region, np.zeros(5))
        #income_adj = np.array([-0.05, -0.05, 0.05, 0.03, 0.02]) if income < 30000 else (np.array([0.05, 0.03, -0.03, -0.03, -0.02]) if income > 80000 else np.zeros(5))
        sex_adj = np.array([-0.05, -0.03, 0.03, 0.03, 0.02]) if sex == 'male' else np.array([0.05, 0.03, -0.03, -0.03, -0.02])

        probs = base_probs + age_adj + region_adj + sex_adj
        probs = np.clip(probs, 0.001, None)
        probs /= probs.sum()
        choice = np.random.choice([1, 2, 3, 4, 5], p=probs)
        result.append(choice)

    return pd.Series(result, index=data.index)


def generate_num_dependents(age_array, marital_status_array, income_array, region_array):
    result = []
    for age, marital_status, income, region in zip(age_array, marital_status_array, income_array, region_array):
        if marital_status == 'married':
            base_mean = 1.8
        elif marital_status == 'divorced':
            base_mean = 1.0
        elif marital_status == 'widowed':
            base_mean = 0.5
        else:
            base_mean = 0.3

        if 25 <= age <= 45:
            age_adj = 0.7
        elif 18 <= age < 25:
            age_adj = 0.3
        elif 46 <= age <= 60:
            age_adj = -0.3
        else:
            age_adj = -0.5

        if income < 30000:
            income_adj = -0.3
        elif income > 100000:
            income_adj = 0.4
        else:
            income_adj = 0.0

        region_adj_map = {
            'north': 0.1,
            'northeast': 0.2,
            'southeast': -0.1,
            'south': -0.1,
            'central_west': 0.0
        }
        region_adj = region_adj_map.get(region, 0.0)

        mean_dependents = base_mean + age_adj + income_adj + region_adj
        mean_dependents = max(mean_dependents, 0)

        num_dependents = np.random.poisson(mean_dependents)
        num_dependents = min(num_dependents, 5)
        result.append(num_dependents)
    return np.array(result)

def generate_num_dependents_v2(df):
    result = []
    for i, (age, marital_status, income, region) in enumerate(zip(df[AGE], df[MARITAL_STATUS], df[INCOME], df[REGION])):
        if marital_status == 'married':
            base_mean = 1.8
        elif marital_status == 'divorced':
            base_mean = 1.0
        elif marital_status == 'widowed':
            base_mean = 0.5
        else:
            base_mean = 0.3

        if 25 <= age <= 45:
            age_adj = 0.7
        elif 18 <= age < 25:
            age_adj = 0.3
        elif 46 <= age <= 60:
            age_adj = -0.3
        else:
            age_adj = -0.5

        if income < 30000:
            income_adj = -0.3
        elif income > 100000:
            income_adj = 0.4
        else:
            income_adj = 0.0

        region_adj_map = {
            'north': 0.1,
            'northeast': 0.2,
            'southeast': -0.1,
            'south': -0.1,
            'central_west': 0.0
        }
        region_adj = region_adj_map.get(region, 0.0)

        mean_dependents = base_mean + age_adj + income_adj + region_adj
        mean_dependents = max(mean_dependents, 0)

        num_dependents = np.random.poisson(mean_dependents)
        num_dependents = min(num_dependents, 5)
        result.append(num_dependents)

    return pd.Series(result, index=df.index)

def generate_num_dependents_kids(df):
    rdm = np.random.default_rng(42)
    result = []
    for i, (age, marital_status, income, region) in enumerate(zip(df[AGE], df[MARITAL_STATUS], df[INCOME], df[REGION])):
        if rdm.random() < 0.45:
            result.append(0)
            continue

        if marital_status == 'married':
            base_mean = 1.0
        elif marital_status == 'divorced':
            base_mean = 1.0
        elif marital_status == 'widowed':
            base_mean = 0.2
        else:
            base_mean = 0.3

        if 25 <= age <= 45:
            age_adj = 0.7
        elif 18 <= age < 25:
            age_adj = 0.3
        elif 46 <= age <= 55:
            if marital_status == 'windowed':
                age_adj = -0.5    
            else:
                age_adj = -0.3
        else:
            age_adj = -2.0
            

        if income < 30000:
            income_adj = -0.3
        elif income > 100000:
            income_adj = 0.4
        else:
            income_adj = 0.2

        region_adj_map = {
            'north': 0.1,
            'northeast': 0.2,
            'southeast': -0.1,
            'south': -0.1,
            'central_west': 0.0
        }
        region_adj = region_adj_map.get(region, 0.0)

        mean_dependents = base_mean + age_adj + income_adj + region_adj
        mean_dependents = max(mean_dependents, 0)

        num_dependents = np.random.poisson(mean_dependents)
        num_dependents = min(num_dependents, 5)
        result.append(num_dependents)

    return pd.Series(result, index=df.index)

def generate_num_dependents_adult(df):
    rdm = np.random.default_rng(42)
    result = []
    for age, marital_status, income, region, kids in zip(df[AGE], df[MARITAL_STATUS], df[INCOME], df[REGION], df[NUM_DEPENDENTS_KIDS]):
        if rdm.random() < 0.8 and kids > 0:
            result.append(0)
            continue

        if marital_status == 'married':
            base_mean = 1.0
        elif marital_status == 'divorced':
            base_mean = 0.1
        elif marital_status == 'widowed':
            base_mean = 0.0
        else:
            base_mean = 0.1
        
        if income < 30000:
            income_adj = -0.2
        elif income > 100000:
            income_adj = 0.1
        else:
            income_adj = 0.0

        region_adj_map = {
            'north': 0.1,
            'northeast': 0.2,
            'southeast': -0.1,
            'south': -0.1,
            'central_west': 0.0
        }
        region_adj = region_adj_map.get(region, 0.0)

        mean_dependents = base_mean + income_adj + region_adj
        mean_dependents = max(mean_dependents, 0)

        num_dependents = np.random.poisson(mean_dependents)
        num_dependents = min(num_dependents, 2)
        result.append(num_dependents)

    return pd.Series(result, index=df.index)

def generate_num_dependents_older(df):
    rdm = np.random.default_rng(42)
    result = []
    for age, marital_status, income, region, kids, adults in zip(df[AGE], df[MARITAL_STATUS], df[INCOME], df[REGION], df[NUM_DEPENDENTS_KIDS], df[NUM_DEPENDENTS_ADULT]):
        if rdm.random() < 0.5:
            if kids > 0 or adults > 0:
                result.append(0)
                continue
            elif kids > 1 or adults > 1:
                result.append(0)
                continue

        if marital_status == 'married':
            base_mean = 0.3
        elif marital_status == 'divorced':
            base_mean = 0.3
        elif marital_status == 'widowed':
            base_mean = 0.1
        else:
            base_mean = 0.2
        
        if income < 30000:
            income_adj = -0.5
        elif income > 100000:
            income_adj = 0.1
        else:
            income_adj = 0.0

        region_adj_map = {
            'north': 0.1,
            'northeast': 0.2,
            'southeast': -0.1,
            'south': -0.1,
            'central_west': 0.0
        }
        region_adj = region_adj_map.get(region, 0.0)

        mean_dependents = base_mean + income_adj + region_adj
        mean_dependents = max(mean_dependents, 0)

        num_dependents = np.random.poisson(mean_dependents)
        num_dependents = min(num_dependents, 2)

        if num_dependents > 2:
            num_dependents = 2

        if age > 60 and num_dependents > 1:
            num_dependents = 1

        result.append(num_dependents)

    return pd.Series(result, index=df.index)

def generate_plan_usage_frequency(df) -> pd.Series:
    usage = []

    for age in df["age"]:
        if age <= 18:
            mean = 1.5
        elif age <= 29:
            mean = 2.0
        elif age <= 39:
            mean = 2.5
        elif age <= 49:
            mean = 3.5
        elif age <= 59:
            mean = 4.5
        elif age <= 69:
            mean = 6.5
        else:
            mean = 8.0

        freq = np.random.poisson(mean)
        usage.append(min(freq, 20)) 

    return pd.Series(usage, index=df.index, name="plan_usage_last_year")

def generate_coverage_level(income_array, age_array):
    result = []
    for income, age in zip(income_array, age_array):
        if income > 100000:
            probs = [0.2, 0.3, 0.5]
        elif income > 50000:
            probs = [0.4, 0.4, 0.2]
        else:
            probs = [0.7, 0.2, 0.1]

        if age > 60:
            probs = [max(probs[0] - 0.1, 0), probs[1], min(probs[2] + 0.1, 1)]
            total = sum(probs)
            probs = [p / total for p in probs]

        choice = np.random.choice(['basic', 'standard', 'premium'], p=probs)
        result.append(choice)
    return np.array(result)

def generate_coverage_level_v2(df):
    result = []
    for i, (income, age) in enumerate(zip(df[INCOME], df[AGE])):
        if income > 100000:
            probs = [0.2, 0.3, 0.5]
        elif income > 50000:
            probs = [0.4, 0.4, 0.2]
        else:
            probs = [0.7, 0.2, 0.1]

        if age > 60:
            probs = [max(probs[0] - 0.1, 0), probs[1], min(probs[2] + 0.1, 1)]
            total = sum(probs)
            probs = [p / total for p in probs]

        choice = np.random.choice(['basic', 'standard', 'premium'], p=probs)
        result.append(choice)

    return pd.Series(result,index=df.index)


def generate_genetic_risk(age_array, sex_array, region_array, smoker_array, bmi_array, alcohol_array, income_array):
    result = []
    for age, sex, region, smoker, bmi, alcohol, income in zip(age_array, sex_array, region_array, smoker_array, bmi_array, alcohol_array, income_array):
        alpha_base = 2
        beta_base = 5

        age_factor = 1.2 if age >= 45 else 0.9
        sex_factor = 1.1 if sex == 'male' else 1.0
        region_factors = {
            'southeast': 1.15,
            'south': 1.12,
            'northeast': 1.05,
            'north': 1.03,
            'central_west': 1.00
        }
        region_factor = region_factors.get(region, 1.0)
        smoker_factor = 1.3 if smoker == 1 else 1.0
        bmi_factor = 1.25 if bmi >= 30 else (1.1 if bmi >= 25 else 1.0)
        alcohol_factor = 1.15 if alcohol > 3 else 1.0
        income_factor = 1.1 if income < 30000 else (0.9 if income > 100000 else 1.0)

        combined_factor = age_factor * sex_factor * region_factor * smoker_factor * bmi_factor * alcohol_factor * income_factor

        alpha_adj = alpha_base * combined_factor
        beta_adj = beta_base

        risk_score = np.random.beta(alpha_adj, beta_adj) * 100
        risk_score = np.clip(risk_score, 0, 100)
        result.append(risk_score)
    return np.array(result)

def generate_genetic_risk_v2(df):
    result = []
    for i, (age, sex, region, smoker, bmi, alcohol, income) in enumerate(zip(df[AGE], df[SEX], df[REGION], df[SMOKER], df[BMI], df[ALCOHOL_CONSUMPTION], df[INCOME])):
        alpha_base = 2
        beta_base = 5

        age_factor = 1.2 if age >= 45 else 0.9
        sex_factor = 1.1 if sex == 'male' else 1.0
        region_factors = {
            'southeast': 1.15,
            'south': 1.12,
            'northeast': 1.05,
            'north': 1.03,
            'central_west': 1.00
        }
        region_factor = region_factors.get(region, 1.0)
        smoker_factor = 1.3 if smoker == 1 else 1.0
        bmi_factor = 1.25 if bmi >= 30 else (1.1 if bmi >= 25 else 1.0)
        alcohol_factor = 1.15 if alcohol > 3 else 1.0
        income_factor = 1.1 if income < 30000 else (0.9 if income > 100000 else 1.0)

        combined_factor = age_factor * sex_factor * region_factor * smoker_factor * bmi_factor * alcohol_factor * income_factor

        alpha_adj = alpha_base * combined_factor
        beta_adj = beta_base

        risk_score = np.random.beta(alpha_adj, beta_adj) * 100
        risk_score = np.clip(risk_score, 0, 100)
        result.append(risk_score)
    
    return pd.Series(result, index=df.index)


def generate_base_data(n_samples):
    #income = np.clip(skewnorm.rvs(-4, loc=36000, scale=18300, size=n_samples), 18300, 500000)
    age = generate_age(n_samples)
    sex = generate_sex(age)
    #education = generate_education(age, income)
    region = np.random.choice(['north', 'northeast', 'southeast', 'south', 'central_west'], n_samples, p=[0.05, 0.15, 0.50, 0.20, 0.10])
    exercise_frequency = generate_exercise_frequency(age, sex)
    bmi = generate_bmi(age, region, exercise_frequency)
    marital_status = generate_marital_status(age)
    smoker = generate_smoker(age, sex)
    chronic_condition = generate_chronic_condition_v2(age)
    #alcohol_consumption = generate_alcohol_consumption(age, sex, income, region)
    #diet_quality = generate_diet_quality(income)
    #occupation_risk = generate_occupation_risk(age, region, income, sex)
    #num_dependents = generate_num_dependents(age, marital_status, income, region)
    #coverage_level = generate_coverage_level(income, age)
    #genetic_risk = generate_genetic_risk(age, sex, region, smoker, bmi, alcohol_consumption, income)

    data = {
        AGE: age,
        SEX: sex,
        REGION: region,
     #   INCOME: income,
        EXERCISE_FREQUENCY: exercise_frequency,
        BMI: bmi,
        MARITAL_STATUS: marital_status,
        SMOKER: smoker,
        CHRONIC_CONDITION: chronic_condition,
     #   ALCOHOL_CONSUMPTION: alcohol_consumption,
     #   DIET_QUALITY: diet_quality,
     #   OCCUPATION_RISK: occupation_risk,
     #   NUM_DEPENDENTS: num_dependents,
     #   COVERAGE_LEVEL: coverage_level,
     #   GENETIC_RISK: genetic_risk,
     #   EDUCATION: education
    }
    return pd.DataFrame(data)


# ==============================================
# UPDATE NUMERICOS PARA CATEGORICOS
# ==============================================

def update_num_to_cat(df):
    df[BMI] = pd.cut(df[BMI],
                       bins=[0, 18.5, 24.9, 29.9, 40],
                       labels=['underweight', 'normal', 'overweight', 'obese'])

    df[ALCOHOL_CONSUMPTION] = pd.cut(df[ALCOHOL_CONSUMPTION],
                                       bins=[-1, 0, 3, 7, 20],
                                       labels=['none', 'low', 'moderate', 'high'])

    df[GENETIC_RISK] = pd.qcut(df[GENETIC_RISK],
                                 q=3,
                                 labels=['low', 'medium', 'high'])

    return df

# ==============================================
# GERAR TARGET (CHARGES)
# ==============================================
def generate_target(df):
    marital_factor = {
        'single': 1.1,
        'married': 1.0,
        'divorced': 1.2,
        'widowed': 1.3
    }

    regional_cost = {
        'north': 1.15,
        'northeast': 1.05,
        'southeast': 0.95,
        'south': 1.05,
        'central_west': 1.0
    }

    coverage_map = {'basic': 1.0, 'standard': 1.2, 'premium': 1.5}

    base = 200

    charges = ((

        base
        + 80 * df[AGE] ** 1.5
        + 5000 * df[SMOKER].fillna(0)  # maior peso para fumantes
        + 1500 * df[CHRONIC_CONDITION].fillna(0) ** 1.5
        + 800 * df[NUM_DEPENDENTS_ADULT].fillna(0) * 12
        + 1500 * df[NUM_DEPENDENTS_OLDER].fillna(0) * 12
        + 15 * np.abs(df[BMI].fillna(25) - 25) ** 0.5
        - 200 * df[EXERCISE_FREQUENCY].map({
            'sedentary': 0,
            'light': 1,
            'moderate': 2,
            'active': 3
        }).fillna(0)
        + 30 * df[ALCOHOL_CONSUMPTION].fillna(0)
        + 0.05 * df[GENETIC_RISK].fillna(df[GENETIC_RISK].median()) * df[AGE]
        + 0.0002 * df[INCOME] * df[REGION].map(regional_cost).fillna(1)  # se quiser manter leve influência de renda
        + 100 * df[OCCUPATION_RISK].fillna(1)
        + 100 * df[USE_LAST_YEAR].fillna(0)

    ) * (1 + 0.55 * df[NUM_DEPENDENTS_KIDS].fillna(0))  # percentual por criança
    )

    charges *= df[COVERAGE_LEVEL].map(coverage_map).fillna(1)
    charges *= df[MARITAL_STATUS].map(marital_factor).fillna(1)

    # Ruído proporcional à idade
    noise = np.random.normal(0, 500 + df[AGE] * 10)
    charges += noise

    return np.clip(charges, 500, 60000).astype(int)

# ==============================================
# VALORES AUSENTES
# ==============================================

def add_missing_values(df):
    missing_percents = {
        GENETIC_RISK: 0.05,
        ALCOHOL_CONSUMPTION: 0.07,
        DIET_QUALITY: 0.10,
        MARITAL_STATUS: 0.03
    }

    for col, pct in missing_percents.items():
        if col in df.columns:
            mask = np.random.rand(len(df)) < pct
            df.loc[mask, col] = np.nan

    return df


# ==============================================
# PLOT DATA
# ==============================================
def plot_health_insurance_data(df):
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))

    # age
    sns.histplot(df[AGE], bins=30, kde=True, ax=axes[0, 0], color='skyblue')
    axes[0, 0].set_title('Distribuição da Idade')
    axes[0, 0].set_xlabel('Idade')

    # SMOKER
    smoker_counts = df[SMOKER].value_counts(normalize=True)
    axes[0, 1].pie(smoker_counts, labels=smoker_counts.index.map({0: 'Não fumante', 1: 'Fumante'}), autopct='%1.1f%%',
                   colors=['#66b3ff', '#ff6666'])
    axes[0, 1].set_title('Proporção de Fumantes')

    # COVERAGE_LEVEL vs CHARGES
    sns.boxplot(x=COVERAGE_LEVEL, y=CHARGES, data=df, ax=axes[1, 0], palette='Set2')
    axes[1, 0].set_title('Distribuição do Custo por Nível de Cobertura')
    axes[1, 0].set_ylabel('Custo (Charges)')
    axes[1, 0].set_xlabel('Nível de Cobertura')

    # BMI vs CHARGES
    sns.scatterplot(x=BMI, y=CHARGES, hue='exercise_frequency', data=df, ax=axes[1, 1], palette='viridis',
                    alpha=0.6)
    axes[1, 1].set_title('Relação entre IMC, Exercício e Custo')
    axes[1, 1].set_xlabel('IMC')
    axes[1, 1].set_ylabel('Custo (Charges)')

    # NUM_DEPENDENTS
    sns.countplot(x=NUM_DEPENDENTS, data=df, ax=axes[2, 0], color='mediumseagreen')
    axes[2, 0].set_title('Número de Dependentes')
    axes[2, 0].set_xlabel('Dependentes')

    # MARITAL_STATUS vs CHARGES
    sns.boxplot(x=MARITAL_STATUS, y=CHARGES, data=df, ax=axes[2, 1], palette='Pastel1')
    axes[2, 1].set_title('Custo por Estado Civil')
    axes[2, 1].set_xlabel('Estado Civil')
    axes[2, 1].set_ylabel('Custo (Charges)')

    plt.tight_layout()
    plt.show()

    # Correlation matrix
    numeric_df = df.select_dtypes(include=['number'])
    corr_matrix = numeric_df.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap='coolwarm',
        cbar=True,
        square=True,
        linewidths=0.5,
        linecolor='gray'
    )
    plt.title('Matriz de Correlação das Variáveis Numéricas')
    plt.tight_layout()
    plt.show()


def print_categorical_distributions(df):
    print("\nDistribuição das variáveis categóricas:")
    for col in [SEX, REGION, MARITAL_STATUS, SMOKER, COVERAGE_LEVEL, EXERCISE_FREQUENCY]:
        print(f"\n{col}:\n", df[col].value_counts(normalize=True))


def print_numeric_distributions(df):
    numeric_cols = df.select_dtypes(include=['number']).columns
    print("\nDistribuição das variáveis numéricas:\n")
    for col in numeric_cols:
        print(f"Variável: {col}")
        print(f"  Média: {df[col].mean():.2f}")
        print(f"  Mediana: {df[col].median():.2f}")
        print(f"  Desvio Padrão: {df[col].std():.2f}")
        print(f"  Mínimo: {df[col].min():.2f}")
        print(f"  25% Quantil: {df[col].quantile(0.25):.2f}")
        print(f"  50% Quantil: {df[col].quantile(0.50):.2f}")
        print(f"  75% Quantil: {df[col].quantile(0.75):.2f}")
        print(f"  Máximo: {df[col].max():.2f}")
        print("-" * 40)


def generate_salubrity(data):
    salubrity = np.zeros(len(data), dtype=int)
    or_5 = data['occupation_risk'] == 5
    or_4 = data['occupation_risk'] == 4
    salubrity[or_5] = (np.random.rand(or_5.sum()) < 0.60).astype(int)
    salubrity[or_4] = (np.random.rand(or_4.sum()) < 0.40).astype(int)

    return pd.Series(salubrity, index=data.index)

def generate_income(df):
    n = len(df)
    income = np.zeros(n)

    minimal_income_montly = 1518
    maximal_income_retired = 8157.41

    for i in range(n):
        age = df.at[i, "age"]
        sex = df.at[i, "sex"]
        occ_risk = df.at[i, "occupation_risk"]
        #salubrity = df.at[i, "salubrity"]

        #regra para dividir o salário entre os jovens, não deixando muitos jovens ricos (herdeiros?)
        if 19 <= age <= 27:
            if np.random.rand() < 0.02:
                income[i] = np.random.uniform(20000, 50000)
            else:
                income[i] = np.random.uniform(minimal_income_montly, 6000)

        #regra de aposentadoria
        elif sex == "male" and age > 65:
            p = np.random.rand()
            if p <= 0.70:
                income[i] = np.random.uniform(1200, minimal_income_montly)
            elif p <= 0.98:
                income[i] = np.random.uniform(minimal_income_montly, maximal_income_retired)
            else:
                income[i] = np.random.uniform(maximal_income_retired, 20000)

        #salario baseado na idade
        else:
            faixa_base = 2000 + (age - 18) * 200
            faixa_limite = faixa_base + 5000
            income[i] = np.random.uniform(faixa_base, faixa_limite)

        #calculo de salubridade
        #if salubrity == 1:
        #    if occ_risk == 4:
        #        income[i] += 282
        #    elif occ_risk == 5:
        #        income[i] += 564

    return pd.Series(np.round(income * 12, 2))


# ==============================================
# PIPELINE PRINCIPAL
# ==============================================

if __name__ == "__main__":
    df = generate_base_data(N_SAMPLES)
    df[OCCUPATION_RISK] = generate_occupation_risk_v2(df)
    #df[SALUBRITY] = generate_salubrity(df)
    df[INCOME] = generate_income(df)
    df[DIET_QUALITY] = generate_diet_quality(df[INCOME])
    df[EXERCISE_FREQUENCY] = generate_exercise_frequency(df[AGE], df[SEX])
    df[BMI] = generate_bmi(df[AGE], df[REGION], df[EXERCISE_FREQUENCY])
    df[EDUCATION] = generate_education_v2(df)
    df[MARITAL_STATUS] = generate_marital_status(df[AGE])
    df[ALCOHOL_CONSUMPTION] = generate_alcohol_consumption_v2(df)
    df[NUM_DEPENDENTS_KIDS] = generate_num_dependents_kids(df)
    df[NUM_DEPENDENTS_ADULT] = generate_num_dependents_adult(df)
    df[NUM_DEPENDENTS_OLDER] = generate_num_dependents_older(df)
    df[COVERAGE_LEVEL] = generate_coverage_level_v2(df)
    df[GENETIC_RISK] = generate_genetic_risk_v2(df)
    df[USE_LAST_YEAR] = generate_plan_usage_frequency(df)
    df[CHARGES] = generate_target(df)
    df = add_missing_values(df)
    df = update_num_to_cat(df)

    df.to_csv('health_insurance_dataset2.csv', index=False)
    print("✅ Dataset gerado com sucesso! Formato:", df.shape)
    print("\n🔍 Amostra do dataset:")
    print(df.sample(3))

    if PLOT:
        print("\n📊 Plot:")
        plot_health_insurance_data(df)

    print("\n📊 Validação Estatística:")
    print("Estatísticas descritivas numéricas:")
    print(df.describe())

    print_categorical_distributions(df)
    print_numeric_distributions(df)
