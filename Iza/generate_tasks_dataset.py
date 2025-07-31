# Importando bibliotecas
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

random.seed(42)
np.random.seed(42)

teams_df = pd.read_csv('/Users/izabela.oliveira/Documents/GitHub/POS-IA/Iza/teams_dataset.csv')


# Parâmetros de horas mínimas e máximas por sprint
min_total_hours = 1600 #quantidade de devs global x quantidade de horas
max_total_hours = 2240

# Multiplicadores usados para calcular esforço/custo no algoritmo
type_multipliers = {
    'Bug': 1.5,
    'Critical Bug': 2.5,
    'Planned Task': 1,
    'Unplanned Task': 1.25,
    'Technical Debt': 1.25,
    'Spike': 1
    #'Disaster': 3
}

priority_multipliers = {
    'Disaster': 2,
    'Critical': 1.5,
    'High': 1,
    'Medium': 0.5,
    'Low': 0.25
}

# Pesos para type
type_generation_weights = {
    'Bug': 0.15, 
    'Critical Bug': 0.05,
    'Planned Task': 0.5,
    'Unplanned Task': 0.15,
    'Technical Debt': 0.10,
    'Spike': 0.05
}

# Pesos para priority
priority_generation_weights = {
    'Disaster': 0.01,
    'Critical': 0.04,
    'High':  0.4,
    'Medium': 0.4,
    'Low': 0.14
}

type_priority_map = {
    #'Disaster': ['Disaster'],
    'Critical Bug': ['Disaster', 'Critical'],
    'Bug': ['Critical', 'High', 'Medium'],
    'Planned Task': ['High', 'Medium', 'Low'],
    'Unplanned Task': ['High', 'Medium', 'Low'],
    'Technical Debt': ['Medium', 'Low'],
    'Spike': ['Medium', 'Low'],
}

# Stacks e pesos
stack_weights = {'BE': 4, 'FE': 2, 'DE': 1, 'DS': 1, 'DB': 1, 'MO': 1}
stacks = list(stack_weights.keys())
weighted_stacks = [s for s, w in stack_weights.items() for _ in range(w)]

stack_type_weights = {
    'BE': {'Bug': 0.15, 'Critical Bug': 0.10, 'Planned Task': 0.50, 'Unplanned Task': 0.10, 'Technical Debt': 0.10, 'Spike': 0.05},
    'FE': {'Bug': 0.15, 'Critical Bug': 0.05, 'Planned Task': 0.65, 'Unplanned Task': 0.05, 'Technical Debt': 0.05, 'Spike': 0.05},
    'DE': {'Bug': 0.15, 'Critical Bug': 0.10, 'Planned Task': 0.50, 'Unplanned Task': 0.10, 'Technical Debt': 0.10, 'Spike': 0.05},
    'DS': {'Bug': 0.03, 'Critical Bug': 0.02, 'Planned Task': 0.80, 'Unplanned Task': 0.10, 'Technical Debt': 0.02, 'Spike': 0.02},
    'DB': {'Bug': 0.05, 'Critical Bug': 0.05, 'Planned Task': 0.60, 'Unplanned Task': 0.10, 'Technical Debt': 0.15, 'Spike': 0.05},
    'MO': {'Bug': 0.10, 'Critical Bug': 0.05, 'Planned Task': 0.70, 'Unplanned Task': 0.05, 'Technical Debt': 0.05, 'Spike': 0.05}
}

# Escala de complexidade
complexity_scale = [1, 2, 3, 5, 8]

# Mapeamento de complexidade para faixa de horas estimadas
hours_per_complexity = {
    1: (2, 4),    # 1 ponto: 2h-4h
    2: (4, 8),    # 2 pontos: 4h-8h
    3: (6, 12),   # 3 pontos: 6h-12h
    5: (10, 20),  # 5 pontos: 10h-20h
    8: (16, 32)   # 8 pontos: 16h-32h
}

stack_hour_multiplier = {
    'BE': 1.0,    # Backend como default
    'FE': 0.85,   # Frontend menos esforço em média
    'DB': 0.90,   # Banco de dados moderado
    'DS': 1.30,   # Data Science demanda mais horas por conta de prototipação e testes
    'DE': 0.95,   # DevOps/Infra na média
    'MO': 1.40    # Mobile com mais tempo por conta da diversidade de dispositivos e testes
}

def generate_task_name(idx):
    return f'Task_{idx+1}'

def create_task(idx, task_type, stack, complexity_points, forced_priority=None):
    min_h, max_h = hours_per_complexity[complexity_points]
    base_hours = random.randint(min_h, max_h)
    adjusted_hours = int(base_hours * stack_hour_multiplier[stack])

    if forced_priority is not None:
        priority = forced_priority
    else:
        possible_priorities = type_priority_map[task_type]
        priority = random.choice(possible_priorities)

    return {
        'name': generate_task_name(idx),
        'type': task_type,
        'type_multiplier': type_multipliers[task_type],
        'priority': priority,
        'priority_multiplier': priority_multipliers[priority],
        'stack': stack,
        'estimated_hours': adjusted_hours,
        'complexity_points': complexity_points
    }

def generate_tasks_balanced_by_team(teams_df):
    tasks = []
    total_hours = 0
    idx = 0

    stacks_list = stacks
    complexity_list = complexity_scale
    types = list(type_generation_weights.keys())
    priorities = list(priority_generation_weights.keys())

    included_types = set()
    included_priorities = set()
    included_stacks = set()
    included_complexities = set()

    # Distribuição proporcional de horas por stack com base na quantidade de devs
    stack_counts = teams_df['stack'].value_counts().to_dict()
    total_devs = sum(stack_counts.values())
    stack_hour_targets = {
        stack: int(min_total_hours * (count / total_devs))
        for stack, count in stack_counts.items()
    }

    stack_current_hours = {stack: 0 for stack in stacks_list}

    def add_task(stack, task_type, complexity_points, forced_priority=None):
        nonlocal total_hours, idx
        task = create_task(idx, task_type, stack, complexity_points, forced_priority)

        if max_total_hours is not None and (total_hours + task['estimated_hours'] > max_total_hours):
            return False
        if stack_current_hours[stack] + task['estimated_hours'] > stack_hour_targets[stack]:
            return False

        tasks.append(task)
        total_hours += task['estimated_hours']
        stack_current_hours[stack] += task['estimated_hours']
        included_types.add(task_type)
        included_priorities.add(task['priority'])
        included_stacks.add(stack)
        included_complexities.add(complexity_points)
        idx += 1
        return True

    # Garantir 1 de cada tipo
    for task_type in types:
        stack = random.choice(list(stack_counts.keys()))
        complexity = random.choice(complexity_list)
        add_task(stack, task_type, complexity)

    # Garantir prioridades ausentes
    missing_priorities = [p for p in priorities if p not in included_priorities]
    for priority in missing_priorities:
        possible_types = [t for t, plist in type_priority_map.items() if priority in plist]
        if not possible_types:
            continue
        task_type = random.choice(possible_types)
        stack = random.choice(list(stack_counts.keys()))
        complexity = random.choice(complexity_list)
        add_task(stack, task_type, complexity, forced_priority=priority)

    # Garantir stacks ausentes
    missing_stacks = [s for s in stacks_list if s not in included_stacks]
    for stack in missing_stacks:
        task_type = random.choices(
            list(stack_type_weights[stack].keys()),
            weights=list(stack_type_weights[stack].values()), k=1
        )[0]
        complexity = random.choice(complexity_list)
        add_task(stack, task_type, complexity)

    # Garantir complexidades ausentes
    missing_complexities = [c for c in complexity_list if c not in included_complexities]
    for complexity in missing_complexities:
        stack = random.choice(list(stack_counts.keys()))
        task_type = random.choices(
            list(stack_type_weights[stack].keys()),
            weights=list(stack_type_weights[stack].values()), k=1
        )[0]
        add_task(stack, task_type, complexity)

    # Preencher até atingir o mínimo de horas
    while total_hours < min_total_hours:
        stack = random.choices(
            list(stack_counts.keys()),
            weights=[stack_counts[s] for s in stack_counts.keys()],
            k=1
        )[0]
        task_type = random.choices(
            list(stack_type_weights[stack].keys()),
            weights=list(stack_type_weights[stack].values()), k=1
        )[0]
        complexity = random.choice(complexity_list)
        added = add_task(stack, task_type, complexity)
        if not added:
            break

    return pd.DataFrame(tasks)




df_tasks = generate_tasks_balanced_by_team(teams_df)
df_tasks.to_csv('/Users/izabela.oliveira/Documents/GitHub/POS-IA/Iza/tasks_dataset_.csv', index=False)

print(df_tasks.groupby(['stack', 'type']).size().unstack(fill_value=0))
print(df_tasks.groupby('stack')['estimated_hours'].sum())
