import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

num_generations = 100
mutation_rate = 0.05
population_size = 50
sprint_days = 10
max_no_improve = 20
no_improve = 0
best_fitness_so_far = float('inf')

random.seed(42)
np.random.seed(42)

best_fitness_history = []
avg_fitness_history = []

level_rules = {
    'Intern':     {'cost_multiplier': 0.5,  'max_daily_hours': 5},
    'Junior':     {'cost_multiplier': 0.75, 'max_daily_hours': 7},
    'Mid-level':  {'cost_multiplier': 1.0,  'max_daily_hours': 7},
    'Senior':     {'cost_multiplier': 1.25, 'max_daily_hours': 7},
    'Specialist': {'cost_multiplier': 1.5,  'max_daily_hours': 7},
}

priority_weight = {'Disaster': 5, 'Critical': 4, 'High':3, 'Medium':2, 'Low':1}

# Carregando datasets
teams_df = pd.read_csv('teams_dataset.csv')
# print("Equipes:")
# print(teams_df.head())

tasks_df = pd.read_csv('tasks_dataset.csv')
# print("\Tasks:")
# print(tasks_df.head())

# print("\nInformações Equipes:")
# print(teams_df.info())

# print("\nInformações Tasks:")
# print(tasks_df.info())

developers = teams_df['name'].tolist()

# dev é compatível
def is_compatible(dev_name, task_stack, teams_df):
    dev_stack = teams_df.loc[teams_df['name'] == dev_name, 'stack'].values[0]
    return dev_stack == task_stack

# geração para criar solução válida com priorização de tasks
def generate_random_chromosome_prioritized(tasks_df, developers, teams_df):
    chromosome = []
    # Ordena tasks da maior para menor prioridade
    tasks_sorted = tasks_df.copy()
    tasks_sorted['priority_weight'] = tasks_sorted['priority'].map(priority_weight)
    tasks_sorted = tasks_sorted.sort_values(by='priority_weight', ascending=False)

    for _, task_row in tasks_sorted.iterrows():
        task_stack = task_row['stack']
        # Devs compatíveis
        compatible_devs = [dev for dev in developers if teams_df.loc[teams_df['name'] == dev, 'stack'].values[0] == task_stack]

        if compatible_devs:
            assigned_dev = random.choice(compatible_devs)
        else: # fallback para qualquer dev
            assigned_dev = random.choice(developers)

        chromosome.append(assigned_dev)

    return chromosome

# gerar população inicial
def generate_random_chromosome_balanced(tasks_df, developers, teams_df):
    dev_hours = {dev: 0 for dev in developers}
    chromosome = []

    for _, task_row in tasks_df.iterrows():
        task_stack = task_row['stack']
        hours = task_row['estimated_hours']
        compatible_devs = [dev for dev in developers if teams_df.loc[teams_df['name']==dev, 'stack'].values[0]==task_stack]
        if compatible_devs:
            # escolhe dev compatível com menor carga atual
            assigned_dev = min(compatible_devs, key=lambda d: dev_hours[d])
        else:
            assigned_dev = min(developers, key=lambda d: dev_hours[d])

        chromosome.append(assigned_dev)
        dev_hours[assigned_dev] += hours

    return chromosome

def generate_population(population_size, tasks_df, developers, teams_df):
    population = []
    half = population_size // 2

    for _ in range(half):
        chromosome = generate_random_chromosome_prioritized(tasks_df, developers, teams_df)
        population.append(chromosome)

    for _ in range(population_size - half):
        chromosome = generate_random_chromosome_balanced(tasks_df, developers, teams_df)
        population.append(chromosome)

    return population

# fitness
def fitness(chromosome, tasks_df, teams_df):
    dev_hours = {dev: 0 for dev in teams_df['name']}
    dev_cost = {dev: 0 for dev in teams_df['name']}
    total_effort = 0
    penalty = 0

    for task_idx, dev in enumerate(chromosome):
        task = tasks_df.iloc[task_idx]
        hours = task['estimated_hours']
        dev_info = teams_df[teams_df['name'] == dev].iloc[0]
        level = dev_info['level']
        stack = dev_info['stack']

        # Acumula horas realizadas pelo dev considerando a sprint toda
        dev_hours[dev] += hours
        total_effort += hours

        # Calcula o custo ponderado pela experiência do dev
        cost = hours * level_rules[level]['cost_multiplier']
        dev_cost[dev] += cost

        # Penalidade se stack do dev difere da stack da tarefa
        if stack != task['stack']:
            penalty += 20

        # Penalidade por sobrecarga
        max_allowed_hours = level_rules[level]['max_daily_hours'] * sprint_days
        if dev_hours[dev] > max_allowed_hours:
            overload = dev_hours[dev] - max_allowed_hours
            penalty += 40 * overload  # penalização severa por sobrecarga

        # Bônus para especialistas e seniors alocando na stack correta (reduz penalidades)
        if (level == 'Specialist' or level == 'Senior') and stack == task['stack']:
            penalty -= 5

    total_cost = sum(dev_cost.values())

    # Fitness final: soma do esforço total (horas), custo, e penalidades — quanto menor melhor
    fitness_value = total_effort + total_cost + penalty

    return fitness_value


# escolhendo os pais por torneio
def tournament_selection(population, fitnesses, k=3):
    selected = random.sample(list(zip(population, fitnesses)), k)
    selected.sort(key=lambda x: x[1])  # Menor fitness é melhor

    return selected[0][0]


# corrigir incompatibilidades
def correct_incompatibilities(chromosome, tasks_df, developers, teams_df):
    corrected_chromosome = chromosome.copy()

    for i, dev in enumerate(chromosome):
        task_stack = tasks_df.iloc[i]['stack']
        dev_stack = teams_df.loc[teams_df['name'] == dev, 'stack'].values[0]

        if dev_stack != task_stack:
            # filtra devs compatíveis com a stack da tarefa
            compatible_devs = [d for d in developers if teams_df.loc[teams_df['name'] == d, 'stack'].values[0] == task_stack]

            if compatible_devs:
                # substitui por um dev compatível aleatório
                corrected_chromosome[i] = random.choice(compatible_devs)
            else:
                # mantém o original se não tem compatível
                pass

    return corrected_chromosome

def uniform_crossover(parent1, parent2):
    child = []

    for a, b in zip(parent1, parent2):
      child.append(random.choice([a, b]))

    return child

# mutação
def mutate_balanced(chromosome, developers, teams_df, tasks_df, mutation_rate=0.05):
    dev_hours = {dev: 0 for dev in developers}

    # calcula a carga atual desse cromossomo
    for i, dev in enumerate(chromosome):
        dev_hours[dev] += tasks_df.iloc[i]['estimated_hours']
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            task_stack = tasks_df.iloc[i]['stack']
            task_hours = tasks_df.iloc[i]['estimated_hours']
            compatible_devs = [dev for dev in developers if teams_df.loc[teams_df['name'] == dev, 'stack'].values[0] == task_stack]
            if compatible_devs:
                # escolhe entre compatibles o de menor carga
                assigned_dev = min(compatible_devs, key=lambda d: dev_hours[d])
                dev_hours[assigned_dev] += task_hours
                dev_hours[chromosome[i]] -= task_hours
                chromosome[i] = assigned_dev
    return chromosome

def validate_chromosome(chromosome, tasks_df, teams_df):
    for i, dev in enumerate(chromosome):
        task_stack = tasks_df.iloc[i]['stack']
        dev_stack = teams_df.loc[teams_df['name'] == dev, 'stack'].values[0]
        if dev_stack != task_stack:
            return False

    return True

def print_dev_load(chromosome, tasks_df):
    dev_hours = defaultdict(float)

    for task_idx, dev in enumerate(chromosome):
        dev_hours[dev] += tasks_df.iloc[task_idx]['estimated_hours']

    print("Carga horária por desenvolvedor:")
    for dev, hours in dev_hours.items():
        print(f"{dev}: {hours:.2f}h")

def check_compatibility(chromosome, tasks_df, teams_df):
    incompatibilities = []
    for i, dev in enumerate(chromosome):
        task_stack = tasks_df.iloc[i]['stack']
        dev_stack = teams_df.loc[teams_df['name'] == dev, 'stack'].values[0]
        if dev_stack != task_stack:
            incompatibilities.append((i, dev, task_stack, dev_stack))
    if incompatibilities:
        print("Incompatibilidades encontradas:")
        for task_idx, dev, task_stack, dev_stack in incompatibilities:
            print(f"Task {task_idx} (stack {task_stack}) -> Dev {dev} (stack {dev_stack})")
    else:
        print("Todas as tasks estão alocadas a devs compatíveis!")


def tasks_per_dev(chromosome):
    count = Counter(chromosome)
    print("Número de tasks por dev:")
    for dev, num in count.items():
        print(f"{dev}: {num}")



population = generate_population(population_size, tasks_df, developers, teams_df)

print(f"População inicial gerada com {population_size} soluções.")
print("Exemplo de cromossomo da população:")
print(population[0])

for generation in range(num_generations):
    # Avaliar fitness de cada indivíduo
    fitness_values = [fitness(individual, tasks_df, teams_df) for individual in population]

    current_best_fitness = min(fitness_values)
    if current_best_fitness < best_fitness_so_far:
        best_fitness_so_far = current_best_fitness
        no_improve = 0
    else:
        no_improve += 1
    if no_improve >= max_no_improve:
        print("Parando por estagnação!")
        break

    new_population = []

    # elitismo
    elite_idx = np.argmin(fitness_values)
    elite_individual = population[elite_idx]
    new_population.append(elite_individual)

    # Criar nova população até completar o tamanho
    while len(new_population) < population_size:
        # Seleção dos pais via torneio (exemplo)
        parent1 = tournament_selection(population, fitness_values, k=3)
        parent2 = tournament_selection(population, fitness_values, k=3)

        # Gerar filho pelo crossover e corrigir incompatibilidades
        child = uniform_crossover(parent1, parent2)
        child = correct_incompatibilities(child, tasks_df, developers, teams_df)

        # Mutação
        child = mutate_balanced(child, developers, teams_df, tasks_df, mutation_rate)
        child = correct_incompatibilities(child, tasks_df, developers, teams_df)

        # Validação do cromossomo
        if validate_chromosome(child, tasks_df, teams_df):
          new_population.append(child)

    population = new_population

    # progresso
    best_fitness = min(fitness_values)
    avg_fitness = sum(fitness_values) / len(fitness_values)
    print(f"Geração {generation + 1} - Melhor fitness: {best_fitness:.2f} - Fitness médio: {avg_fitness:.2f}")

    best_fitness_history.append(best_fitness)
    avg_fitness_history.append(avg_fitness)

# Ao final pode pegar o melhor indivíduo para análise
fitness_values = [fitness(ind, tasks_df, teams_df) for ind in population]
best_idx = np.argmin(fitness_values)
best_solution = population[best_idx]
print("Melhor solução encontrada:")
print(best_solution)

# Análises complementares
print_dev_load(best_solution, tasks_df)
check_compatibility(best_solution, tasks_df, teams_df)
tasks_per_dev(best_solution)

plt.figure(figsize=(10,6))
plt.plot(best_fitness_history, label='Melhor Fitness')
plt.plot(avg_fitness_history, label='Fitness Médio')
plt.xlabel('Geração')
plt.ylabel('Fitness')
plt.title('Evolução do Fitness ao Longo das Gerações')
plt.legend()
plt.tight_layout()
plt.show()