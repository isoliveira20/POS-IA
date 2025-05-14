import pandas as pd
import numpy as np
import random

# Configurações iniciais
np.random.seed(42)
num_records = 50_000

def generate_synthetic_data():
    data = {}

    # === Idade e Gênero ===
    data['age'] = np.clip(np.random.normal(35, 15, num_records).astype(int), 5, 95)
    data['gender'] = np.random.choice(['M', 'F'], size=num_records)

    # === Estado Civil (correlacionado com idade) ===
    def get_marital_status(age):
        if age < 25:
            return np.random.choice(['Solteiro(a)', 'Casado(a)', 'Divorciado(a)', 'Viúvo(a)'], p=[0.85, 0.13, 0.01, 0.01])
        elif age < 35:
            return np.random.choice(['Solteiro(a)', 'Casado(a)', 'Divorciado(a)', 'Viúvo(a)'], p=[0.45, 0.50, 0.04, 0.01])
        elif age < 50:
            return np.random.choice(['Solteiro(a)', 'Casado(a)', 'Divorciado(a)', 'Viúvo(a)'], p=[0.20, 0.65, 0.13, 0.02])
        elif age < 65:
            return np.random.choice(['Solteiro(a)', 'Casado(a)', 'Divorciado(a)', 'Viúvo(a)'], p=[0.10, 0.60, 0.20, 0.10])
        else:
            return np.random.choice(['Solteiro(a)', 'Casado(a)', 'Divorciado(a)', 'Viúvo(a)'], p=[0.05, 0.45, 0.15, 0.35])
    
    data['marital_status'] = [get_marital_status(age) for age in data['age']]

    # === Educação (correlacionada com idade) ===
    education_levels = ['Ensino Fundamental', 'Ensino Médio', 'Graduação', 'Pós-graduação', 'Mestrado', 'Doutorado']
    
    def get_education_level(age):
        if age < 25:
            return np.random.choice(education_levels, p=[0.10, 0.50, 0.35, 0.03, 0.01, 0.01])
        elif age < 35:
            return np.random.choice(education_levels, p=[0.05, 0.35, 0.40, 0.12, 0.06, 0.02])
        elif age < 50:
            return np.random.choice(education_levels, p=[0.15, 0.30, 0.30, 0.15, 0.07, 0.03])
        else:
            return np.random.choice(education_levels, p=[0.25, 0.35, 0.20, 0.10, 0.07, 0.03])
    
    data['education_level'] = [get_education_level(age) for age in data['age']]

    # === Ocupação (correlacionada com educação) ===
    occupation_choices = {
        'Ensino Fundamental': ['Operário', 'Auxiliar de Serviços Gerais', 'Atendente', 'Autônomo', 'Vendedor'],
        'Ensino Médio': ['Vendedor', 'Técnico', 'Assistente Administrativo', 'Motorista', 'Recepcionista'],
        'Graduação': ['Analista', 'Professor', 'Engenheiro', 'Contador', 'Enfermeiro'],
        'Pós-graduação': ['Gerente', 'Coordenador', 'Consultor', 'Especialista', 'Administrador'],
        'Mestrado': ['Gerente Sênior', 'Pesquisador', 'Professor Universitário', 'Engenheiro Especializado', 'Médico'],
        'Doutorado': ['Pesquisador Sênior', 'Professor Universitário', 'Médico Especialista', 'Cientista', 'Executivo']
    }
    data['occupation'] = [
        np.random.choice(occupation_choices[edu])
        for edu in data['education_level']
    ]

    # === Dependentes (correlacionados com idade e estado civil) ===
    def get_dependents(age, marital):
        max_dep = max(0, min(5, (age - 25) // 5)) if age >= 25 else 0
        if max_dep == 0:
            return 0
        if marital == 'Solteiro(a)':
            probs = [0.85] + [(0.15 / max_dep)] * max_dep
        elif marital == 'Casado(a)':
            probs = [0.2] + [(0.8 / max_dep)] * max_dep
        else:
            probs = [0.4] + [(0.6 / max_dep)] * max_dep
        return np.random.choice(range(max_dep + 1), p=probs)
    
    data['number_dependents'] = [
        get_dependents(age, marital)
        for age, marital in zip(data['age'], data['marital_status'])
    ]

    # === Renda Anual (baseada em idade, educação, ocupação) ===
    base_income = np.random.lognormal(10.5, 0.5, num_records)
    
    edu_multipliers = {'Ensino Fundamental': 0.6, 'Ensino Médio': 0.8, 'Graduação': 1.0, 'Pós-graduação': 1.3, 'Mestrado': 1.5, 'Doutorado': 1.8}
    occ_multipliers = {
        'baixa_renda': 0.7, 'media_renda': 1.0,
        'alta_renda': 1.5, 'muito_alta_renda': 2.0
    }
    occ_categories = {
        'baixa_renda': ['Operário', 'Auxiliar de Serviços Gerais', 'Atendente', 'Motorista', 'Recepcionista'],
        'media_renda': ['Vendedor', 'Técnico', 'Assistente Administrativo', 'Autônomo', 'Enfermeiro'],
        'alta_renda': ['Analista', 'Professor', 'Contador', 'Coordenador', 'Consultor'],
        'muito_alta_renda': ['Gerente', 'Engenheiro', 'Especialista', 'Administrador', 'Pesquisador', 'Médico', 
                             'Gerente Sênior', 'Professor Universitário', 'Engenheiro Especializado', 'Pesquisador Sênior',
                             'Médico Especialista', 'Cientista', 'Executivo']
    }

    def get_occ_category(occ):
        for cat, occs in occ_categories.items():
            if occ in occs:
                return cat
        return 'media_renda'

    data['annual_income'] = []
    for i in range(num_records):
        age = data['age'][i]
        edu = data['education_level'][i]
        occ = data['occupation'][i]
        age_factor = min(1.5, (age / 30) if age <= 50 else (50 / 30))
        income = base_income[i] * edu_multipliers[edu] * occ_multipliers[get_occ_category(occ)] * age_factor
        income = max(14400, min(round(income * 500) / 100, 500_000))
        data['annual_income'].append(income)


    # === Localização (correlacionada com renda) ===
    locations = ['Centro', 'Zona Norte', 'Zona Sul', 'Zona Leste', 'Zona Oeste']
    location_probs = [0.25, 0.2, 0.2, 0.15, 0.2]
    data['location'] = np.random.choice(locations, size=num_records, p=location_probs)
    # === Tipo de apólice (correlacionado com renda e localização) ===
    policy_types = ['Saúde', 'Vida', 'Automóvel', 'Residencial', 'Viagem']

    def get_policy_type(income, location):
        if income < 30_000:
            return np.random.choice(policy_types[:2], p=[0.6, 0.4])
        elif income < 100_000:
            return np.random.choice(policy_types, p=[0.4, 0.3, 0.2, 0.05, 0.05])
        else:
            return np.random.choice(policy_types, p=[0.2, 0.3, 0.25, 0.15, 0.1])

    data['policy_type'] = [
        get_policy_type(income, loc)
        for income, loc in zip(data['annual_income'], data['location'])
    ]
    # === Data de início da apólice (aleatória) ===
    start_dates = pd.date_range(start='2010-01-01', end='2023-01-01', freq='D')
    data['policy_start_date'] = np.random.choice(start_dates, size=num_records)
    # === Status de fumante (correlacionado com renda e idade) ===
    smoking_status = ['Não Fumante', 'Fumante']
    def get_smoking_status(income, age):
        if income < 30_000:
            return np.random.choice(smoking_status, p=[0.9, 0.1])
        elif income < 100_000:
            return np.random.choice(smoking_status, p=[0.7, 0.3])
        else:
            return np.random.choice(smoking_status, p=[0.5, 0.5])
    data['smoking_status'] = [
        get_smoking_status(income, age)
        for income, age in zip(data['annual_income'], data['age'])
    ]
    # === Frequência de exercícios (correlacionada com idade e renda) ===
    exercise_freq = ['Nunca', 'Raramente', 'Ocasionalmente', 'Frequentemente', 'Sempre']
    def get_exercise_freq(income, age):
        if income < 30_000:
            return np.random.choice(exercise_freq, p=[0.5, 0.3, 0.1, 0.05, 0.05])
        elif income < 100_000:
            return np.random.choice(exercise_freq, p=[0.2, 0.25, 0.25, 0.15, 0.15])
        else:
            return np.random.choice(exercise_freq, p=[0.1, 0.15, 0.25, 0.25, 0.25])
    data['exercise_frequency'] = [
        get_exercise_freq(income, age)
        for income, age in zip(data['annual_income'], data['age'])
    ]
    # === Tipo de propriedade (correlacionado com renda e localização) ===
    property_types = ['Casa', 'Apartamento', 'Cobertura', 'Chácara', 'Sítio']
    def get_property_type(income, location):
        if income < 30_000:
            return np.random.choice(property_types[:2], p=[0.7, 0.3])
        elif income < 100_000:
            return np.random.choice(property_types, p=[0.4, 0.3, 0.2, 0.05, 0.05])
        else:
            return np.random.choice(property_types, p=[0.2, 0.25, 0.25, 0.15, 0.15])
    data['property_type'] = [
        get_property_type(income, loc)
        for income, loc in zip(data['annual_income'], data['location'])
    ]
    # === Pontuação de saúde (correlacionada com idade, renda e frequência de exercícios) ===
    def get_health_score(age, income, exercise):
        base_score = 100 - (age - 20) * 0.5 + (income / 100_000) * 10
        if exercise == 'Nunca':
            base_score -= 20
        elif exercise == 'Raramente':
            base_score -= 10
        elif exercise == 'Ocasionalmente':
            base_score += 0
        elif exercise == 'Frequentemente':
            base_score += 10
        else:
            base_score += 20
        return max(0, min(100, round(base_score)))
    data['health_score'] = [

        get_health_score(age, income, exercise)
        for age, income, exercise in zip(data['age'], data['annual_income'], data['exercise_frequency'])
    ]
    # === Valor do prêmio (correlacionado com renda, idade, saúde e tipo de apólice) ===
    def get_premium_amount(income, age, health, policy):
        base_premium = (income / 1000) + (age / 10) - (health / 10)
        if policy == 'Saúde':
            base_premium *= 1.5
        elif policy == 'Vida':
            base_premium *= 1.2
        elif policy == 'Automóvel':
            base_premium *= 1.3
        elif policy == 'Residencial':
            base_premium *= 1.1
        else:
            base_premium *= 0.9
        return max(500, min(round(base_premium), 50_000))
    data['premium_amount'] = [
        get_premium_amount(income, age, health, policy)
        for income, age, health, policy in zip(data['annual_income'], data['age'], data['health_score'], data['policy_type'])
    ]
    
    return pd.DataFrame(data)

# Exemplo de uso:
df = generate_synthetic_data()
print(df.head())
df.to_csv('insurance_dataset.csv', index=False)