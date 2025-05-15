import pandas as pd
import numpy as np
from scipy.stats import skewnorm, truncnorm

np.random.seed(42)
N_SAMPLES = 40_000
MISSING_PERCENT = 0.05

def generate_base_data(n_samples):
    income = np.clip(skewnorm.rvs(2, loc=3500, scale=3000, size=n_samples), 1518, 50000)
    age = truncnorm((0 - 35) / 20, (90 - 35) / 20, loc=35, scale=20).rvs(n_samples).astype(int)

    diet_quality = np.where(
        income > 15000,
        np.random.choice([3, 4, 5], size=n_samples, p=[0.1, 0.4, 0.5]),
        np.random.choice([1, 2, 3], size=n_samples, p=[0.4, 0.4, 0.2])
    )

    exercise_frequency = np.empty(n_samples, dtype=object)
# Faixa etária < 30
    mask1 = age < 30

    exercise_frequency[mask1] = np.random.choice(
    ['active', 'moderate', 'light'],
    size=mask1.sum(),
    p=[0.4, 0.4, 0.2]
)

# Faixa etária entre 30 e 60
    mask2 = (age >= 30) & (age < 60)
    exercise_frequency[mask2] = np.random.choice(
    ['active','moderate', 'light', 'sedentary'],
    size=mask2.sum(),
    p=[0.3, 0.2, 0.2, 0.3]
)

# Faixa etária ≥ 60
    mask3 = age >= 60
    exercise_frequency[mask3] = np.random.choice(
    ['moderate', 'sedentary', 'light'],
    size=mask3.sum(),
    p=[0.1, 0.6, 0.3]
)

    chronic_conditions = np.random.poisson(0.5 + (age / 60) + (income < 5000).astype(int), size=n_samples)
    chronic_conditions = np.clip(chronic_conditions, 0, 5)

    base_bmi = 25 + np.random.normal(0, 3, size=n_samples)
    bmi = base_bmi \
        + (age - 40) * 0.05 \
        + np.select(
            [diet_quality == 1, diet_quality == 2, diet_quality == 3, diet_quality == 4, diet_quality == 5],
            [2.5, 1.5, 0, -1, -2], default=0
        ) \
        + pd.Series(exercise_frequency).map({'sedentary': 2, 'light': 1, 'moderate': -1, 'active': -2}).values
    bmi = np.clip(bmi, 16, 40)

    data = {
        'age': age,
        'sex': np.random.choice(['male', 'female'], n_samples, p=[0.49, 0.51]),
        'region': np.random.choice(['north', 'south', 'east', 'west'], n_samples, p=[0.35, 0.15, 0.35, 0.15]),
        'bmi': bmi,
        'smoker': np.random.binomial(1, 0.2, n_samples),
        'chronic_conditions': chronic_conditions,
        'genetic_risk': np.random.beta(2, 5, n_samples) * 100,
        'exercise_frequency': exercise_frequency,
        'alcohol_consumption': np.clip(np.random.poisson(2, n_samples), 0, 20),
        'diet_quality': diet_quality,
        'income': income,
        'occupation_risk': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.4, 0.3, 0.15, 0.1, 0.05]),
        'coverage_level': np.random.choice(['basic', 'standard', 'premium'], n_samples, p=[0.6, 0.25, 0.15]),
        'num_dependents': np.random.poisson(1.2, n_samples)
        }
    return pd.DataFrame(data)
    
def add_engineered_features(df):
    older_mask = df['age'] > 60
    df.loc[older_mask, 'coverage_level'] = np.random.choice(
        ['basic', 'standard', 'premium'],
        size=older_mask.sum(),
        p=[0.1, 0.3, 0.6]
    )

    regional_cost = {'north': 1.15, 'south': 0.95, 'east': 1.05, 'west': 0.95}
    coverage_map = {'basic': 1.0, 'standard': 1.2, 'premium': 1.5}

    df['regional_cost_factor'] = df['region'].map(regional_cost)
    df['coverage_factor'] = df['coverage_level'].map(coverage_map)

    return df

def generate_target(df):
    base = 200
    charges = (
        base
        + 80 * df['age'] ** 1.5
        + 3000 * df['smoker']
        + 1500 * df['chronic_conditions'] * (1 + 0.5 * df['smoker'])
        + 20 * np.abs(df['bmi'] - 25) ** 2.5
        - 300 * df['exercise_frequency'].map({
            'sedentary': 0,
            'light': 1,
            'moderate': 2,
            'active': 3
        })
        + 80 * df['alcohol_consumption'] ** 1.3
        + 0.08 * df['genetic_risk'] * df['age']
        + 0.0003 * df['income'] * df['regional_cost_factor']
        + 30 * df['occupation_risk']
        + 50 * df['num_dependents']
        + 0.00005 * df['income'] * (6 - df['diet_quality'])
    )

    charges *= df['coverage_factor']
    noise = np.random.normal(0, 400 + df['age'] * 8)
    charges += noise

    return np.clip(charges, 1518, 18216).astype(int)

def add_missing_values(df, missing_percent):
    cols_with_missing = ['genetic_risk', 'alcohol_consumption', 'diet_quality']
    for col in cols_with_missing:
        mask = np.random.rand(len(df)) < missing_percent
        df.loc[mask, col] = np.nan
    return df

def normalize_column_names(df):
    return df.rename(columns={
        'exercise_frequency': 'exercise',
        'diet_quality': 'diet',
        'num_dependents': 'dependents',
    })

if __name__ == "__main__":
    df = generate_base_data(N_SAMPLES)
    df = add_engineered_features(df)
    df['charges'] = generate_target(df)
    df = add_missing_values(df, MISSING_PERCENT)
    df_model = normalize_column_names(df)

    df_model.to_csv('health_insurance_dataset.csv', index=False)
    print("✅ Dataset gerado com sucesso! Formato:", df_model.shape)
    print("\n🔍 Amostra do dataset:")
    print(df_model.sample(3))
