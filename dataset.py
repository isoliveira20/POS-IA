import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Configuração
np.random.seed(42)  # Para reprodutibilidade
num_records = 200000  # Pode ser alterado para um número menor durante testes

# Definição de parâmetros com correlações planejadas
def generate_synthetic_data():
    data = {}
    
    # Parâmetros básicos
    data['age'] = np.random.normal(40, 15, num_records).astype(int)
    # Ajustar idades para o intervalo realista (18-90)
    data['age'] = np.clip(data['age'], 18, 90)
    
    # Gênero (com distribuição aproximadamente 50/50)
    data['gender'] = np.random.choice(['M', 'F'], size=num_records)
    
    # Renda anual fortemente correlacionada com idade e nível de educação (será definido depois)
    # Base para renda
    base_income = np.random.lognormal(10.5, 0.5, num_records)
    
    # Estado civil correlacionado com idade
    marital_choices = ['Solteiro(a)', 'Casado(a)', 'Divorciado(a)', 'Viúvo(a)']
    marital_status = []
    
    for age in data['age']:
        if age < 25:
            probs = [0.85, 0.13, 0.01, 0.01]
        elif age < 35:
            probs = [0.45, 0.50, 0.04, 0.01]
        elif age < 50:
            probs = [0.20, 0.65, 0.13, 0.02]
        elif age < 65:
            probs = [0.10, 0.60, 0.20, 0.10]
        else:
            probs = [0.05, 0.45, 0.15, 0.35]
        marital_status.append(np.random.choice(marital_choices, p=probs))
        
    data['marital_status'] = marital_status
    
    # Número de dependentes - correlacionado com idade e estado civil
    dependents = []
    for age, marital in zip(data['age'], data['marital_status']):
        max_dependents = max(0, min(5, (age - 25) // 5)) if age >= 25 else 0
        
        if marital == 'Solteiro(a)':
            # Solteiros têm menos probabilidade de ter dependentes
            # Corrigido problema de tamanho do array:
            if max_dependents == 0:
                dep = 0  # Se max_dependents é 0, a única opção é 0 dependentes
            else:
                # Calculamos probabilidades corretamente para o range específico 
                probs = [0.85] + [(0.15/max_dependents)] * max_dependents
                dep = np.random.choice(range(max_dependents+1), p=probs)
        elif marital == 'Casado(a)':
            # Casados têm mais probabilidade de ter dependentes
            if max_dependents == 0:
                dep = 0
            else:
                probs = [0.2] + [(0.8/max_dependents)] * max_dependents
                dep = np.random.choice(range(max_dependents+1), p=probs)
        else:
            # Divorciados e viúvos têm probabilidade média
            if max_dependents == 0:
                dep = 0
            else:
                probs = [0.4] + [(0.6/max_dependents)] * max_dependents
                dep = np.random.choice(range(max_dependents+1), p=probs)
                
        dependents.append(dep)
    
    data['number_dependents'] = dependents
    
    # Nível de educação - correlacionado com idade
    education_choices = ['Ensino Fundamental', 'Ensino Médio', 'Graduação', 'Pós-graduação', 'Mestrado', 'Doutorado']
    education_level = []
    
    for age in data['age']:
        if age < 25:
            probs = [0.10, 0.50, 0.35, 0.03, 0.01, 0.01]
        elif age < 35:
            probs = [0.05, 0.35, 0.40, 0.12, 0.06, 0.02]
        elif age < 50:
            probs = [0.15, 0.30, 0.30, 0.15, 0.07, 0.03]
        else:
            probs = [0.25, 0.35, 0.20, 0.10, 0.07, 0.03]
        education_level.append(np.random.choice(education_choices, p=probs))
    
    data['education_level'] = education_level
    
    # Ocupação - correlacionada com nível de educação
    occupation_choices = {
        'Ensino Fundamental': ['Operário', 'Auxiliar de Serviços Gerais', 'Atendente', 'Autônomo', 'Vendedor'],
        'Ensino Médio': ['Vendedor', 'Técnico', 'Assistente Administrativo', 'Motorista', 'Recepcionista'],
        'Graduação': ['Analista', 'Professor', 'Engenheiro', 'Contador', 'Enfermeiro'],
        'Pós-graduação': ['Gerente', 'Coordenador', 'Consultor', 'Especialista', 'Administrador'],
        'Mestrado': ['Gerente Sênior', 'Pesquisador', 'Professor Universitário', 'Engenheiro Especializado', 'Médico'],
        'Doutorado': ['Pesquisador Sênior', 'Professor Universitário', 'Médico Especialista', 'Cientista', 'Executivo']
    }
    
    data['occupation'] = [np.random.choice(occupation_choices[edu]) for edu in data['education_level']]
    
    # Ajustar renda baseada em educação e ocupação
    income = []
    
    education_multipliers = {
        'Ensino Fundamental': 0.6,
        'Ensino Médio': 0.8,
        'Graduação': 1.0,
        'Pós-graduação': 1.3,
        'Mestrado': 1.5,
        'Doutorado': 1.8
    }
    
    occupation_categories = {
        'baixa_renda': ['Operário', 'Auxiliar de Serviços Gerais', 'Atendente', 'Motorista', 'Recepcionista'],
        'media_renda': ['Vendedor', 'Técnico', 'Assistente Administrativo', 'Autônomo', 'Enfermeiro'],
        'alta_renda': ['Analista', 'Professor', 'Contador', 'Coordenador', 'Consultor'],
        'muito_alta_renda': ['Gerente', 'Engenheiro', 'Especialista', 'Administrador', 'Pesquisador', 'Médico', 'Gerente Sênior', 'Professor Universitário', 'Engenheiro Especializado', 'Pesquisador Sênior', 'Médico Especialista', 'Cientista', 'Executivo']
    }
    
    occupation_multipliers = {
        'baixa_renda': 0.7,
        'media_renda': 1.0,
        'alta_renda': 1.5,
        'muito_alta_renda': 2.0
    }
    
    def get_occupation_category(occupation):
        for category, occupations in occupation_categories.items():
            if occupation in occupations:
                return category
        return 'media_renda'  # padrão
    
    for i, (base_inc, edu, occ, age) in enumerate(zip(base_income, data['education_level'], data['occupation'], data['age'])):
        # Fator idade: aumenta até ~50 anos, depois estabiliza
        age_factor = min(1.5, (age / 30) if age <= 50 else (50 / 30))
        
        # Multiplicadores
        edu_mult = education_multipliers[edu]
        occ_category = get_occupation_category(occ)
        occ_mult = occupation_multipliers[occ_category]
        
        # Calcular renda final
        final_income = base_inc * edu_mult * occ_mult * age_factor
        
        # Ajustar para valores realistas em reais
        final_income = round(final_income * 500) / 100
        final_income = min(final_income, 500000)  # Limitar valores extremos
        final_income = max(final_income, 14400)   # Garantir pelo menos salário mínimo mensal x 12
        
        income.append(final_income)
    
    data['annual_income'] = income
    
    # Score de saúde - correlacionado com idade, renda, e estilo de vida (que será definido depois)
    base_health = np.random.normal(70, 15, num_records)
    
    # Tabagismo (definir agora para usar no score de saúde)
    smoking_probabilities = []
    for edu in data['education_level']:
        if edu in ['Ensino Fundamental', 'Ensino Médio']:
            smoking_probabilities.append(0.3)
        elif edu in ['Graduação', 'Pós-graduação']:
            smoking_probabilities.append(0.2)
        else:
            smoking_probabilities.append(0.1)
    
    data['smoking_status'] = np.random.binomial(1, smoking_probabilities)
    data['smoking_status'] = ['Sim' if s == 1 else 'Não' for s in data['smoking_status']]
    
    # Frequência de exercícios - correlacionada com idade e educação
    exercise_choices = ['Nunca', 'Raramente', 'Ocasionalmente', 'Regularmente', 'Diariamente']
    exercise_frequency = []
    
    for age, edu in zip(data['age'], data['education_level']):
        if age < 30:
            if edu in ['Ensino Fundamental', 'Ensino Médio']:
                probs = [0.15, 0.25, 0.30, 0.20, 0.10]
            else:
                probs = [0.10, 0.15, 0.25, 0.30, 0.20]
        elif age < 50:
            if edu in ['Ensino Fundamental', 'Ensino Médio']:
                probs = [0.20, 0.30, 0.25, 0.15, 0.10]
            else:
                probs = [0.15, 0.20, 0.25, 0.25, 0.15]
        else:
            if edu in ['Ensino Fundamental', 'Ensino Médio']:
                probs = [0.25, 0.30, 0.25, 0.15, 0.05]
            else:
                probs = [0.20, 0.25, 0.25, 0.20, 0.10]
        
        exercise_frequency.append(np.random.choice(exercise_choices, p=probs))
    
    data['exercise_frequency'] = exercise_frequency
    
    # Calcular score de saúde final baseado nos fatores
    health_scores = []
    
    exercise_health_bonus = {
        'Nunca': -10,
        'Raramente': -5,
        'Ocasionalmente': 0,
        'Regularmente': 5,
        'Diariamente': 10
    }
    
    for i, base_h in enumerate(base_health):
        # Ajustes de idade (-0.5 pontos por ano acima de 30)
        age_adjustment = max(-25, min(0, -(data['age'][i] - 30) * 0.5)) if data['age'][i] > 30 else 0
        
        # Ajuste de tabagismo
        smoking_adjustment = -15 if data['smoking_status'][i] == 'Sim' else 0
        
        # Ajuste de exercício
        exercise_adjustment = exercise_health_bonus[data['exercise_frequency'][i]]
        
        # Ajuste de renda (assumindo que maior renda = melhor acesso a cuidados de saúde)
        income_percentile = np.searchsorted(np.sort(data['annual_income']), data['annual_income'][i]) / num_records
        income_adjustment = 10 * (income_percentile - 0.5)  # -5 to +5
        
        # Score final de saúde
        final_health = base_h + age_adjustment + smoking_adjustment + exercise_adjustment + income_adjustment
        final_health = max(1, min(100, round(final_health)))  # Garantir intervalo 1-100
        
        health_scores.append(final_health)
        
    data['health_score'] = health_scores
    
    # Local (regiões brasileiras)
    cities_by_region = {
        'Norte': ['Manaus', 'Belém', 'Porto Velho', 'Rio Branco', 'Macapá', 'Boa Vista', 'Palmas'],
        'Nordeste': ['Salvador', 'Recife', 'Fortaleza', 'São Luís', 'João Pessoa', 'Maceió', 'Natal', 'Aracaju', 'Teresina'],
        'Centro-Oeste': ['Brasília', 'Goiânia', 'Cuiabá', 'Campo Grande'],
        'Sudeste': ['São Paulo', 'Rio de Janeiro', 'Belo Horizonte', 'Vitória'],
        'Sul': ['Curitiba', 'Florianópolis', 'Porto Alegre']
    }
    
    # Distribuição da população por região aproximadamente
    region_probabilities = {
        'Sudeste': 0.42,
        'Nordeste': 0.27,
        'Sul': 0.14,
        'Norte': 0.09,
        'Centro-Oeste': 0.08
    }
    
    regions = np.random.choice(
        list(region_probabilities.keys()),
        size=num_records,
        p=list(region_probabilities.values())
    )
    
    locations = []
    for region in regions:
        locations.append(np.random.choice(cities_by_region[region]))
    
    data['location'] = locations
    
    # Tipo de apólice - correlacionado com idade, renda, e estado civil
    policy_types = []
    
    for age, income, marital in zip(data['age'], data['annual_income'], data['marital_status']):
        income_percentile = np.searchsorted(np.sort(data['annual_income']), income) / num_records
        
        if age < 30:
            if marital == 'Solteiro(a)':
                if income_percentile < 0.3:
                    probs = [0.7, 0.2, 0.1, 0.0]  # Básica, Padrão, Premium, VIP
                elif income_percentile < 0.7:
                    probs = [0.3, 0.5, 0.2, 0.0]
                else:
                    probs = [0.1, 0.5, 0.3, 0.1]
            else:
                if income_percentile < 0.3:
                    probs = [0.5, 0.3, 0.2, 0.0]
                elif income_percentile < 0.7:
                    probs = [0.2, 0.5, 0.3, 0.0]
                else:
                    probs = [0.1, 0.3, 0.5, 0.1]
        elif age < 50:
            if income_percentile < 0.3:
                probs = [0.4, 0.4, 0.2, 0.0]
            elif income_percentile < 0.7:
                probs = [0.2, 0.4, 0.3, 0.1]
            else:
                probs = [0.1, 0.2, 0.5, 0.2]
        else:
            if income_percentile < 0.3:
                probs = [0.3, 0.4, 0.2, 0.1]
            elif income_percentile < 0.7:
                probs = [0.1, 0.3, 0.5, 0.1]
            else:
                probs = [0.0, 0.2, 0.5, 0.3]
                
        policy_types.append(np.random.choice(['Básica', 'Padrão', 'Premium', 'VIP'], p=probs))
    
    data['policy_type'] = policy_types
    
    # Sinistro anterior - correlacionado com idade, score de saúde, e tipo de apólice
    previous_claims = []
    
    for age, health, policy in zip(data['age'], data['health_score'], data['policy_type']):
        base_prob = 0.1  # Probabilidade base
        
        # Ajuste idade (mais velho = maior probabilidade)
        age_factor = age / 50
        
        # Ajuste saúde (menos saudável = maior probabilidade)
        health_factor = (100 - health) / 50
        
        # Ajuste tipo de apólice (melhor apólice = mais provável ter histórico)
        policy_factor = {
            'Básica': 0.6,
            'Padrão': 0.8,
            'Premium': 1.0,
            'VIP': 1.2
        }[policy]
        
        # Probabilidade final
        final_prob = min(0.9, base_prob * age_factor * health_factor * policy_factor)
        
        previous_claims.append('Sim' if np.random.random() < final_prob else 'Não')
    
    data['previous_claim'] = previous_claims
    
    # Idade do veículo - correlacionada com renda
    vehicle_ages = []
    
    for income in data['annual_income']:
        income_percentile = np.searchsorted(np.sort(data['annual_income']), income) / num_records
        
        # Quanto maior a renda, mais novo o veículo tende a ser
        if income_percentile < 0.2:
            mean_age = 12
        elif income_percentile < 0.4:
            mean_age = 9
        elif income_percentile < 0.6:
            mean_age = 7
        elif income_percentile < 0.8:
            mean_age = 5
        else:
            mean_age = 3
            
        # Gerar idade do veículo com distribuição normal
        vehicle_age = max(0, int(np.random.normal(mean_age, 2)))
        vehicle_ages.append(vehicle_age)
    
    data['vehicle_age'] = vehicle_ages
    
    # Score de crédito - correlacionado com renda, idade, educação
    credit_scores = []
    
    for income, age, edu in zip(data['annual_income'], data['age'], data['education_level']):
        income_percentile = np.searchsorted(np.sort(data['annual_income']), income) / num_records
        
        # Base para score de crédito
        base_score = 500 + 300 * income_percentile
        
        # Ajuste de idade (mais velho = melhor histórico)
        age_factor = min(1.2, max(0.8, age / 40))
        
        # Ajuste de educação
        edu_factor = {
            'Ensino Fundamental': 0.9,
            'Ensino Médio': 0.95,
            'Graduação': 1.0,
            'Pós-graduação': 1.05,
            'Mestrado': 1.07,
            'Doutorado': 1.1
        }[edu]
        
        # Score final
        final_score = min(900, int(base_score * age_factor * edu_factor))
        credit_scores.append(final_score)
    
    data['credit_score'] = credit_scores
    
    # Duração do seguro - correlacionada com idade e tipo de apólice
    insurance_durations = []
    
    for age, policy in zip(data['age'], data['policy_type']):
        if age < 30:
            if policy in ['Básica', 'Padrão']:
                mean_duration = 1.5
            else:
                mean_duration = 2.5
        elif age < 50:
            if policy in ['Básica', 'Padrão']:
                mean_duration = 3
            else:
                mean_duration = 5
        else:
            if policy in ['Básica', 'Padrão']:
                mean_duration = 5
            else:
                mean_duration = 8
                
        duration = max(1, int(np.random.normal(mean_duration, 1)))
        insurance_durations.append(duration)
    
    data['insurance_duration'] = insurance_durations
    
    # Data de início da apólice - últimos 5 anos
    policy_start_dates = []
    
    end_date = datetime(2025, 1, 1)
    start_date = end_date - timedelta(days=365*5)  # 5 anos atrás
    days_range = (end_date - start_date).days
    
    for _ in range(num_records):
        random_days = np.random.randint(0, days_range)
        date = start_date + timedelta(days=random_days)
        policy_start_dates.append(date.strftime('%Y-%m-%d'))
    
    data['policy_start_date'] = policy_start_dates
    
    # Tipo de propriedade - correlacionado com renda, estado civil, e número de dependentes
    property_types = []
    
    for income, marital, dependents in zip(data['annual_income'], data['marital_status'], data['number_dependents']):
        income_percentile = np.searchsorted(np.sort(data['annual_income']), income) / num_records
        
        if marital == 'Solteiro(a)' and dependents == 0:
            if income_percentile < 0.3:
                probs = [0.7, 0.2, 0.1, 0.0]  # Apartamento, Casa, Casa de Luxo, Mansão
            elif income_percentile < 0.7:
                probs = [0.6, 0.3, 0.1, 0.0]
            else:
                probs = [0.5, 0.3, 0.2, 0.0]
        elif dependents <= 1:
            if income_percentile < 0.3:
                probs = [0.5, 0.4, 0.1, 0.0]
            elif income_percentile < 0.7:
                probs = [0.4, 0.5, 0.1, 0.0]
            else:
                probs = [0.3, 0.5, 0.2, 0.0]
        else:
            if income_percentile < 0.3:
                probs = [0.3, 0.6, 0.1, 0.0]
            elif income_percentile < 0.7:
                probs = [0.2, 0.6, 0.2, 0.0]
            else:
                probs = [0.1, 0.5, 0.3, 0.1]
                
        property_types.append(np.random.choice(['Apartamento', 'Casa', 'Casa de Luxo', 'Mansão'], p=probs))
    
    data['property_type'] = property_types
    
    # Valor do prêmio - correlacionado com tipo de apólice, idade, score de saúde, sinistros anteriores
    premium_amounts = []
    
    base_premiums = {
        'Básica': 1000,
        'Padrão': 2000,
        'Premium': 4000,
        'VIP': 8000
    }
    
    for policy, age, health, claim in zip(data['policy_type'], data['age'], data['health_score'], data['previous_claim']):
        base = base_premiums[policy]
        
        # Ajuste de idade (mais velho = maior prêmio)
        if age < 25:
            age_factor = 1.3
        elif age < 35:
            age_factor = 1.0
        elif age < 50:
            age_factor = 1.1
        elif age < 65:
            age_factor = 1.3
        else:
            age_factor = 1.5
            
        # Ajuste de saúde (pior saúde = maior prêmio)
        health_factor = 1 + ((100 - health) / 200)
        
        # Ajuste de sinistro anterior
        claim_factor = 1.5 if claim == 'Sim' else 1.0
        
        # Calcular prêmio final
        premium = base * age_factor * health_factor * claim_factor
        
        # Adicionar aleatoriedade (±10%)
        premium = premium * np.random.uniform(0.9, 1.1)
        
        # Arredondar para número inteiro
        premium = round(premium)
        
        premium_amounts.append(premium)
    
    data['premium_amount'] = premium_amounts
    
    return pd.DataFrame(data)

# Gerar os dados (pode ajustar para um número menor para testes)
# Por exemplo, use num_records = 1000 para testes iniciais
df = generate_synthetic_data()

# Salvar como CSV
df.to_csv('human_data_synthetic_200k.csv', index=False)

# Mostrar algumas estatísticas e as primeiras linhas
print(f"Shape do DataFrame: {df.shape}")
print("\nPrimeiras 10 linhas:")
print(df.head(10))

print("\nResumo estatístico:")
print(df.describe())

# Verificar correlações
print("\nMatriz de correlação (para variáveis numéricas):")
numeric_columns = ['age', 'annual_income', 'number_dependents', 'health_score', 
                  'vehicle_age', 'credit_score', 'insurance_duration', 'premium_amount']
correlation_matrix = df[numeric_columns].corr()
print(correlation_matrix)

# Para demonstrar a saída do CSV sem gerar todo o arquivo, mostrar algumas linhas
print("\nExemplo de como os dados CSV ficarão:")
print(df.head().to_csv(index=False))