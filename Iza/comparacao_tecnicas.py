# Script completo com comparações automáticas de técnicas genéticas

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import time
from collections import defaultdict, Counter
from itertools import product

# Carregar os dados
tasks_df = pd.read_csv('/Users/izabela.oliveira/Documents/GitHub/POS-IA/Iza/tasks_dataset.csv')
teams_df = pd.read_csv('/Users/izabela.oliveira/Documents/GitHub/POS-IA/Iza/teams_dataset.csv')

# Parâmetros gerais
num_generations = 100
mutation_rate = 0.05
population_size = len(tasks_df)
sprint_days = 10
max_no_improve = 20

random.seed(42)
np.random.seed(42)

# Regras fictícias de exemplo (substitua pelas suas reais)
level_rules = {
    'Intern':     {'cost_multiplier': 0.5,  'max_daily_hours': 5},
    'Junior':     {'cost_multiplier': 0.75, 'max_daily_hours': 7},
    'Mid-level':  {'cost_multiplier': 1.0,  'max_daily_hours': 7},
    'Senior':     {'cost_multiplier': 1.25, 'max_daily_hours': 7},
    'Specialist': {'cost_multiplier': 1.5,  'max_daily_hours': 7},
}

priority_weight = {'Disaster': 5, 'Critical': 4, 'High':3, 'Medium':2, 'Low':1}

# Obter lista de desenvolvedores
developers = teams_df['name'].tolist()

# Funções auxiliares



def correct_incompatibilities(chromosome, tasks_df, developers, teams_df):
    for i, dev in enumerate(chromosome):
        task_stack = tasks_df.iloc[i]['stack']
        dev_stacks = teams_df[teams_df['name'] == dev]['stack'].values[0].split(',')
        if task_stack not in dev_stacks:
            compatible_devs = teams_df[teams_df['stack'].str.contains(task_stack)]['name'].tolist()
            if compatible_devs:
                chromosome[i] = random.choice(compatible_devs)
    return chromosome

def validate_chromosome(chromosome, tasks_df, teams_df):
    for i, dev in enumerate(chromosome):
        task_stack = tasks_df.iloc[i]['stack']
        dev_stacks = teams_df[teams_df['name'] == dev]['stack'].values[0].split(',')
        if task_stack not in dev_stacks:
            return False
    return True

def mutate_balanced(chromosome, developers, teams_df, tasks_df, mutation_rate):
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            stack = tasks_df.iloc[i]['stack']
            compatible_devs = teams_df[teams_df['stack'].str.contains(stack)]['name'].tolist()
            if compatible_devs:
                chromosome[i] = random.choice(compatible_devs)
    return chromosome

def uniform_crossover(p1, p2):
    return [random.choice([a, b]) for a, b in zip(p1, p2)]

def one_point_crossover(p1, p2):
    point = random.randint(1, len(p1) - 1)
    return p1[:point] + p2[point:]

def swap_mutation(chromosome):
    a, b = random.sample(range(len(chromosome)), 2)
    chromosome[a], chromosome[b] = chromosome[b], chromosome[a]
    return chromosome

def tournament_selection(population, fitnesses, k=3):
    selected = random.sample(list(zip(population, fitnesses)), k)
    selected.sort(key=lambda x: x[1])
    return selected[0][0]

def roulette_selection(population, fitnesses):
    inverse_fitness = [1.0 / (f + 1e-6) for f in fitnesses]  # evitar divisão por zero
    total = sum(inverse_fitness)
    probs = [f / total for f in inverse_fitness]
    return population[np.random.choice(len(population), p=probs)]

def select_parents(method, population, fitnesses):
    return tournament_selection(population, fitnesses) if method == 'tournament' else roulette_selection(population, fitnesses)

def crossover(method, p1, p2):
    return uniform_crossover(p1, p2) if method == 'uniform' else one_point_crossover(p1, p2)

def mutate(method, chrom, devs, teams, tasks, rate):
    return mutate_balanced(chrom, devs, teams, tasks, rate) if method == 'balanced' else swap_mutation(chrom)

def run_experiment(selection_method, crossover_method, mutation_method, use_elitism, experiment_id):
    population = generate_population(population_size, tasks_df, developers, teams_df)
    best_fitness_so_far = float('inf')
    no_improve = 0
    best_fitness_history = []
    avg_fitness_history = []
    start_time = time.time()

    for generation in range(num_generations):
        fitnesses = [fitness(ind, tasks_df, teams_df) for ind in population]
        best = min(fitnesses)
        avg = sum(fitnesses) / len(fitnesses)
        best_fitness_history.append(best)
        avg_fitness_history.append(avg)

        if best < best_fitness_so_far:
            best_fitness_so_far = best
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= max_no_improve:
            break

        new_population = []
        if use_elitism:
            elite = population[np.argmin(fitnesses)]
            new_population.append(elite)

        while len(new_population) < population_size:
            p1 = select_parents(selection_method, population, fitnesses)
            p2 = select_parents(selection_method, population, fitnesses)
            child = crossover(crossover_method, p1, p2)
            child = correct_incompatibilities(child, tasks_df, developers, teams_df)
            child = mutate(mutation_method, child, developers, teams_df, tasks_df, mutation_rate)
            child = correct_incompatibilities(child, tasks_df, developers, teams_df)
            if validate_chromosome(child, tasks_df, teams_df):
                new_population.append(child)

        population = new_population

    total_time = time.time() - start_time
    return {
        'experiment_id': experiment_id,
        'selection': selection_method,
        'crossover': crossover_method,
        'mutation': mutation_method,
        'elitism': use_elitism,
        'best_fitness': best_fitness_so_far,
        'avg_fitness': avg_fitness_history[-1],
        'generations': generation + 1,
        'time': total_time,
        'fitness_history': best_fitness_history
    }

# Executar experimentos
SELECTIONS = ['tournament', 'roulette']
CROSSOVERS = ['uniform', 'one_point']
MUTATIONS = ['balanced', 'swap']
ELITISM = [True, False]

EXPERIMENTS = list(product(SELECTIONS, CROSSOVERS, MUTATIONS, ELITISM))
results = []

for i, (sel, cross, mut, elit) in enumerate(EXPERIMENTS):
    print(f"Executando experimento {i+1}/{len(EXPERIMENTS)}: {sel}, {cross}, {mut}, elit={elit}")
    result = run_experiment(sel, cross, mut, elit, experiment_id=i+1)
    results.append(result)

# Análise dos resultados
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by='best_fitness')
print(results_df[['experiment_id', 'selection', 'crossover', 'mutation', 'elitism', 'best_fitness', 'avg_fitness', 'generations', 'time']])

# Plotar históricos
plt.figure(figsize=(10, 6))
for result in results:
    plt.plot(result['fitness_history'], label=f"Exp {result['experiment_id']}")
plt.title("Evolução do Fitness por Experimento")
plt.xlabel("Geração")
plt.ylabel("Fitness")
plt.legend()
plt.show()
