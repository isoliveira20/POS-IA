import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import numpy as np

# Configuração de seed para reprodutibilidade (opcional)
random.seed(42)
np.random.seed(42)

# Pesos para experiência (ajustados para ser mais realista)
experience_weights = {
    'Intern': 0.10,      # 8% - poucos estagiários
    'Junior': 0.20,      # 15% - entrada no mercado
    'Mid-level': 0.35,   # 45% - maior parte da força de trabalho
    'Senior': 0.25,      # 25% - experiência sólida
    'Specialist': 0.10   # 7% - especialistas raros
}

experience_levels = list(experience_weights.keys())

# Configuração das equipes
teams_info = [('A', 8), ('B', 7), ('C', 6), ('D', 5), ('E', 4), ('F', 3)]

# Stacks e pesos (BE como maioria)
stacks = ['FE', 'BE', 'DE', 'DS', 'DB', 'MO']
stack_weights = {
    'BE': 50,    # Backend - 50% do pool
    'FE': 15,    # Frontend - necessário  
    'DE': 5,     # DevOps/Engineering
    'DS': 10,     # Data Science
    'DB': 5,     # Database specialist
    'MO': 15      # Mobile
}

# Ordem hierárquica dos níveis
level_order = {
    'Intern': 0,
    'Junior': 1,
    'Mid-level': 2,
    'Senior': 3,
    'Specialist': 4
}

# Pontuação de experiência para balanceamento
experience_points = {
    'Intern': 1,
    'Junior': 2,
    'Mid-level': 4,
    'Senior': 6,
    'Specialist': 8
}

def weighted_choice(weight_dict):
    """Escolha ponderada de um item baseado nos pesos"""
    items = list(weight_dict.keys())
    weights = list(weight_dict.values())
    return random.choices(items, weights=weights, k=1)[0]

def calculate_team_requirements(size):
    """Calcula requisitos ideais para uma equipe baseado no tamanho"""
    # Calcular quantos BEs precisamos (sempre maioria, mas com limite para times pequenos)
    if size < 3:
        # Times pequenos: mínimo 1 BE, máximo 2 BE
        min_be_devs = 1
        max_be_devs = 2
    
    if size >= 3:
        # Times maiores: maioria BE
        max_be_devs = size // 2 + 1

    if size >= 6:
        # Times pequenos: mínimo 1 BE, máximo 2 BE
        min_be_devs = 2
        max_be_devs = 4  
   
    else:
        # Times maiores: maioria normal
        min_be_devs = (size // 2)  # Mais da metade
        max_be_devs = min(size - 1, int(size * 0.50))  # Máximo 70%, deixando espaço para outras stacks
    
    requirements = {
        'min_stacks': max(2, min(6, size - 1)),  # Entre 2-6 stacks diferentes
        'max_specialists': 1 if size <= 5 else 2,
        'max_seniors': max(1, size // 3),  # ~33% podem ser seniors
        'max_interns': 1 if size > 6 else 0,  # Só times maiores têm estagiários
        'max_juniors': max(2, size // 10),  # ~25% podem ser juniors
        'min_experience_points': size * 2.5,  # Experiência mínima por equipe
        'ideal_experience_points': size * 4,   # Experiência ideal por equipe
        'min_be_devs': min_be_devs,  # BE deve ser maioria (com limite para times pequenos)
        'max_be_devs': max_be_devs,  # Limite máximo de BE
        'max_de_devs': 1 if size <= 5 else 2,  # DevOps/Engineering
    }
    
    # Stacks obrigatórias baseadas no tamanho da equipe
    requirements['required_stacks'] = ['BE']  # BE sempre obrigatório
    
    # Times pequenos (2-4 pessoas): foco no essencial
    if size >= 2:
        requirements['required_stacks'].append('FE')  # Frontend essencial
    
    # Times médios (5+ pessoas): começar diversificação
    if size >= 5:
        requirements['required_stacks'].extend(['DE', 'MO'])  # DevOps + Mobile
    
    # Times grandes (7+ pessoas): máxima diversidade
    if size >= 7:
        requirements['required_stacks'].extend(['DS', 'DB'])  # Data Science + Database
    
    return requirements

def generate_balanced_devs(teams_info):
    """Gera desenvolvedores considerando as necessidades de todas as equipes"""
    total_devs = sum(size for _, size in teams_info)
    
    # Calcular necessidades agregadas de todas as equipes
    total_requirements = {
        'required_stacks': set(),
        'total_specialists_needed': 0,
        'total_seniors_needed': 0,
        'min_experience_needed': 0
    }
    
    for team_name, size in teams_info:
        req = calculate_team_requirements(size)
        total_requirements['required_stacks'].update(req['required_stacks'])
        total_requirements['total_specialists_needed'] += req['max_specialists']
        total_requirements['total_seniors_needed'] += req['max_seniors']
        total_requirements['min_experience_needed'] += req['min_experience_points']
    
    devs = []
    counts_level = defaultdict(int)
    counts_stack = defaultdict(int)
    specialist_stacks = set()
    current_experience_total = 0
    
    # Primeira passagem: garantir stacks essenciais e BE como maioria
    essential_devs = 0
    
    # Para cada time, garantir pelo menos a quantidade mínima de BE
    total_be_needed = 0
    for team_name, size in teams_info:
        req = calculate_team_requirements(size)
        total_be_needed += req['min_be_devs']
    
    # Gerar BEs primeiro (maioria garantida)
    be_devs_created = 0
    levels_for_be = ['Senior', 'Mid-level', 'Senior', 'Mid-level', 'Specialist']  # Priorizar experiência para BE
    
    while be_devs_created < total_be_needed and essential_devs < total_devs:
        level = levels_for_be[be_devs_created % len(levels_for_be)]
        dev = {
            'dev_id': f'Dev{essential_devs + 1}',
            'level': level,
            'stack': 'BE'
        }
        devs.append(dev)
        counts_level[level] += 1
        counts_stack['BE'] += 1
        current_experience_total += experience_points[level]
        essential_devs += 1
        be_devs_created += 1
    
    # Depois garantir outras stacks essenciais
    other_essential_stacks = total_requirements['required_stacks'] - {'BE'}
    for stack in other_essential_stacks:
        if essential_devs >= total_devs:
            break
        level = 'Mid-level'  # Nível seguro para outras stacks
        dev = {
            'dev_id': f'Dev{essential_devs + 1}',
            'level': level,
            'stack': stack
        }
        devs.append(dev)
        counts_level[level] += 1
        counts_stack[stack] += 1
        current_experience_total += experience_points[level]
        essential_devs += 1
    
    # Segunda passagem: preencher o restante com balanceamento inteligente
    for i in range(essential_devs + 1, total_devs + 1):
        # Calcular se precisamos de mais experiência
        remaining_devs = total_devs - i + 1
        experience_needed = total_requirements['min_experience_needed'] - current_experience_total
        avg_experience_needed = experience_needed / max(1, remaining_devs)
        
        # Ajustar pesos baseado na necessidade
        adjusted_weights = experience_weights.copy()
        
        # Se precisamos de mais experiência, aumentar peso de níveis seniores
        if avg_experience_needed > 3:
            adjusted_weights['Senior'] *= 2
            adjusted_weights['Specialist'] *= 1.5
            adjusted_weights['Mid-level'] *= 1.2
            adjusted_weights['Intern'] *= 0.5
            adjusted_weights['Junior'] *= 0.7
        
        # Controlar limites máximos
        if counts_level['Intern'] >= total_devs * 0.05:  # Max 8% estagiários
            adjusted_weights.pop('Intern', None)
        if counts_level['Junior'] >= total_devs * 0.20:  # Max 20% juniors
            adjusted_weights.pop('Junior', None)
        if counts_level['Specialist'] >= total_requirements['total_specialists_needed']:
            adjusted_weights.pop('Specialist', None)
        if counts_level['Senior'] >= total_requirements['total_seniors_needed']:
            adjusted_weights['Senior'] *= 0.35
        
        if not adjusted_weights:
            adjusted_weights = {'Mid-level': 1}
        
        level = weighted_choice(adjusted_weights)
        
        # Escolher stack baseado na necessidade e diversidade
        stack_scores = {}
        for stack in stacks:
            base_score = stack_weights[stack]
            
            # SUPER BONUS para BE se ainda não atingiu a maioria necessária
            if stack == 'BE':
                be_percentage = counts_stack['BE'] / max(1, i - 1)
                if be_percentage < 0.55:  # Se BE está abaixo de 55%
                    base_score *= 5  # Multiplicador alto para priorizar BE
                elif be_percentage < 0.50:  # Se BE está abaixo de 50%
                    base_score *= 10  # Multiplicador muito alto
            
            # Bonus para stacks com poucos desenvolvedores (exceto BE que tem lógica própria)
            if stack != 'BE' and counts_stack[stack] < total_devs * 0.10:  # Menos de 10%
                base_score *= 1.5
            
            # Penalty para stacks não-BE com muitos desenvolvedores
            if stack != 'BE' and counts_stack[stack] > total_devs * 0.25:  # Mais de 25%
                base_score *= 0.2
            
            # Specialists precisam de stacks únicas
            if level == 'Specialist' and stack in specialist_stacks:
                base_score = 0
            
            stack_scores[stack] = max(1, base_score)
        
        stack = weighted_choice(stack_scores)
        
        # Registrar specialist stack
        if level == 'Specialist':
            specialist_stacks.add(stack)
        
        dev = {
            'dev_id': f'Dev{i}',
            'level': level,
            'stack': stack
        }
        devs.append(dev)
        counts_level[level] += 1
        counts_stack[stack] += 1
        current_experience_total += experience_points[level]
    
    return pd.DataFrame(devs)

def distribute_devs_to_balanced_teams(df_devs, teams_info):
    """Distribui desenvolvedores para equipes de forma equilibrada e inteligente"""
    teams = []
    dev_pool = df_devs.copy()
    dev_pool['assigned'] = False
    
    def calculate_team_balance(team_devs):
        """Calcula métricas de balanceamento da equipe"""
        if not team_devs:
            return 0
        
        levels = [d['level'] for d in team_devs]
        stacks = [d['stack'] for d in team_devs]
        
        # Pontuação de experiência
        exp_score = sum(experience_points[level] for level in levels)
        
        # Diversidade de stacks
        stack_diversity = len(set(stacks))
        
        # Penalizar desequilíbrios
        penalties = 0
        level_counts = Counter(levels)
        
        # Muitos iniciantes
        if level_counts['Intern'] + level_counts['Junior'] > len(team_devs) * 0.5:
            penalties += 40
        
        # Muito poucos seniores em times grandes
        if len(team_devs) >= 5 and level_counts['Senior'] + level_counts['Specialist'] == 0:
            penalties += 15
        
        return exp_score + stack_diversity * 5 - penalties
    
    def select_best_dev_for_team(team_devs, team_size, available_devs):
        """Seleciona o melhor desenvolvedor para a equipe baseado em múltiplos critérios"""
        if available_devs.empty:
            return None
        
        requirements = calculate_team_requirements(team_size)
        current_stacks = [d['stack'] for d in team_devs]
        current_levels = [d['level'] for d in team_devs]
        level_counts = Counter(current_levels)
        stack_counts = Counter(current_stacks)
        
        # Verificar quantos BEs já temos
        current_be_count = stack_counts.get('BE', 0)
        
        best_dev = None
        best_score = -1000
        
        for idx, dev in available_devs.iterrows():
            score = 0
            
            # PRIORIDADE MÁXIMA: Garantir maioria BE (com limite para times pequenos)
            if dev['stack'] == 'BE':
                if current_be_count < requirements['min_be_devs']:
                    score += 1000  # Score altíssimo para BE quando necessário
                elif current_be_count < requirements['max_be_devs']:
                    score += 100   # Score alto para BE adicional (respeitando limite)
                else:
                    score -= 200   # Penalizar BE em excesso (mais forte para times pequenos)
            else:
                # Para não-BE, verificar se BE já está garantido
                if current_be_count < requirements['min_be_devs']:
                    score -= 500  # Penalizar muito não-BE se BE não está garantido
                elif current_be_count >= requirements['max_be_devs']:
                    score += 50   # Bonus para não-BE se BE já atingiu o máximo
            
            # Pontuação base de experiência
            score += experience_points[dev['level']] * 2
            
            # Bonus para stacks obrigatórias que ainda não temos
            if dev['stack'] in requirements['required_stacks'] and dev['stack'] not in current_stacks:
                score += 50
            
            # Bonus para diversidade de stack (mas menor que BE)
            if dev['stack'] not in current_stacks and dev['stack'] != 'BE':
                score += 20
            
            # Controlar limites por nível
            if dev['level'] == 'Specialist' and level_counts['Specialist'] >= requirements['max_specialists']:
                continue
            if dev['level'] == 'Senior' and level_counts['Senior'] >= requirements['max_seniors']:
                score -= 5
            if dev['level'] == 'Intern' and level_counts['Intern'] >= requirements['max_interns']:
                continue
            if dev['level'] == 'Junior' and level_counts['Junior'] >= requirements['max_juniors']:
                continue
            
            # Bonus para stacks com alta demanda (menor para não confundir com prioridade BE)
            if dev['stack'] != 'BE':
                score += stack_weights.get(dev['stack'], 1) * 0.05
            
            # Pequena aleatoriedade para evitar determinismo total
            score += random.uniform(-1, 1)
            
            if score > best_score:
                best_score = score
                best_dev = idx
        
        return best_dev
    
    # Distribuir desenvolvedores para cada equipe
    for team_name, size in sorted(teams_info, key=lambda x: -x[1]):  # Times maiores primeiro
        team_devs = []
        available_devs = dev_pool[dev_pool['assigned'] == False]
        
        # Selecionar desenvolvedores um por um
        for position in range(size):
            available_devs = dev_pool[dev_pool['assigned'] == False]
            
            if available_devs.empty:
                break
            
            best_dev_idx = select_best_dev_for_team(team_devs, size, available_devs)
            
            if best_dev_idx is None:
                # Fallback: escolher aleatoriamente
                best_dev_idx = random.choice(available_devs.index.tolist())
            
            chosen_dev = dev_pool.loc[best_dev_idx]
            dev_pool.at[best_dev_idx, 'assigned'] = True
            
            team_devs.append({
                'name': f"{team_name}_Dev{len(team_devs)+1}",
                'team': team_name,
                'level': chosen_dev['level'],
                'stack': chosen_dev['stack']
            })
        
        teams.extend(team_devs)
    
    return pd.DataFrame(teams)

def analyze_team_composition(df_teams):
    """Analisa e exibe estatísticas da composição das equipes"""
    print("=== ANÁLISE DE COMPOSIÇÃO DAS EQUIPES ===\n")
    
    # Estatísticas por equipe
    for team in sorted(df_teams['team'].unique()):
        team_data = df_teams[df_teams['team'] == team]
        print(f"EQUIPE {team} ({len(team_data)} membros):")
        
        # Distribuição por nível
        level_dist = team_data['level'].value_counts()
        print(f"  Níveis: {dict(level_dist)}")
        
        # Distribuição por stack
        stack_dist = team_data['stack'].value_counts()
        print(f"  Stacks: {dict(stack_dist)}")
        
        # Verificar maioria BE (com lógica especial para times pequenos)
        be_count = stack_dist.get('BE', 0)
        total_team = len(team_data)
        be_percentage = (be_count / total_team) * 100
        
        if total_team <= 3:
            # Para times pequenos: verificar se respeitou o limite máximo
            is_valid = be_count <= 2
            status = f"✅ DENTRO DO LIMITE (máx 2)" if is_valid else f"❌ EXCEDEU LIMITE (máx 2)"
        else:
            # Para times maiores: verificar maioria
            is_majority = be_count > total_team // 2
            status = "✅ MAIORIA" if is_majority else "❌ NÃO É MAIORIA"
        
        print(f"  BE: {be_count}/{total_team} ({be_percentage:.1f}%) - {status}")
        
        # Pontuação de experiência
        exp_score = sum(experience_points[level] for level in team_data['level'])
        avg_exp = exp_score / len(team_data)
        print(f"  Experiência: {exp_score} pontos (média: {avg_exp:.1f})")
        
        print()
    
    # Estatísticas globais
    print("=== ESTATÍSTICAS GLOBAIS ===")
    print(f"Total de desenvolvedores: {len(df_teams)}")
    
    # Verificar se BE é maioria globalmente
    global_be = len(df_teams[df_teams['stack'] == 'BE'])
    global_total = len(df_teams)
    global_be_pct = (global_be / global_total) * 100
    print(f"BE Global: {global_be}/{global_total} ({global_be_pct:.1f}%)")
    
    print(f"\nDistribuição por nível:")
    for level, count in df_teams['level'].value_counts().items():
        pct = (count / len(df_teams)) * 100
        print(f"  {level}: {count} ({pct:.1f}%)")
    
    print(f"\nDistribuição por stack:")
    for stack, count in df_teams['stack'].value_counts().items():
        pct = (count / len(df_teams)) * 100
        print(f"  {stack}: {count} ({pct:.1f}%)")
    
    # Verificar quantas equipes seguem as regras de BE
    teams_following_rules = 0
    total_teams = len(df_teams['team'].unique())
    
    for team in df_teams['team'].unique():
        team_data = df_teams[df_teams['team'] == team]
        be_count = len(team_data[team_data['stack'] == 'BE'])
        team_size = len(team_data)
        
        if team_size <= 3:
            # Para times pequenos: máximo 2 BE
            follows_rule = be_count <= 2
        else:
            # Para times maiores: maioria BE
            follows_rule = be_count > team_size // 2
            
        if follows_rule:
            teams_following_rules += 1
    
    print(f"\n📊 RESUMO BE:")
    print(f"  Equipes seguindo regras BE: {teams_following_rules}/{total_teams}")
    print(f"  Percentual de sucesso: {(teams_following_rules/total_teams)*100:.1f}%")
    print(f"  Regra: Times ≤3 pessoas = máx 2 BE | Times >3 pessoas = maioria BE")

def create_visualization(df_teams):
    """Cria visualizações da distribuição das equipes"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Distribuição de níveis por equipe
    level_by_team = pd.crosstab(df_teams['team'], df_teams['level'])
    level_by_team.plot(kind='bar', stacked=True, ax=axes[0,0], colormap='viridis')
    axes[0,0].set_title('Distribuição de Níveis por Equipe')
    axes[0,0].set_xlabel('Equipe')
    axes[0,0].set_ylabel('Número de Desenvolvedores')
    axes[0,0].legend(title='Nível')
    axes[0,0].tick_params(axis='x', rotation=0)
    
    # 2. Distribuição de stacks por equipe
    stack_by_team = pd.crosstab(df_teams['team'], df_teams['stack'])
    stack_by_team.plot(kind='bar', stacked=True, ax=axes[0,1], colormap='tab10')
    axes[0,1].set_title('Distribuição de Stacks por Equipe')
    axes[0,1].set_xlabel('Equipe')
    axes[0,1].set_ylabel('Número de Desenvolvedores')
    axes[0,1].legend(title='Stack')
    axes[0,1].tick_params(axis='x', rotation=0)
    
    # 3. Heatmap de experiência por equipe
    experience_data = []
    for team in sorted(df_teams['team'].unique()):
        team_data = df_teams[df_teams['team'] == team]
        total_exp = sum(experience_points[level] for level in team_data['level'])
        avg_exp = total_exp / len(team_data)
        experience_data.append([team, len(team_data), total_exp, avg_exp])
    
    exp_df = pd.DataFrame(experience_data, columns=['Team', 'Size', 'Total_Exp', 'Avg_Exp'])
    
    # Criar matriz para heatmap
    heatmap_data = exp_df.pivot_table(values='Avg_Exp', index=['Team'], aggfunc='mean')
    sns.heatmap(heatmap_data, annot=True, fmt='.1f', ax=axes[1,0], cmap='YlOrRd')
    axes[1,0].set_title('Experiência Média por Equipe')
    
    # 4. Distribuição global
    global_dist = df_teams['level'].value_counts()
    axes[1,1].pie(global_dist.values, labels=global_dist.index, autopct='%1.1f%%')
    axes[1,1].set_title('Distribuição Global de Níveis')
    
    plt.tight_layout()
    plt.show()

# Executar o sistema completo
print("Gerando desenvolvedores balanceados...")
df_devs = generate_balanced_devs(teams_info)

print("Distribuindo para equipes...")
df_teams = distribute_devs_to_balanced_teams(df_devs, teams_info)

# Exibir resultados
print("\n" + "="*50)
print("DESENVOLVEDORES GERADOS:")
print("="*50)
print(df_devs.groupby(['level', 'stack']).size().unstack(fill_value=0))

print("\n" + "="*50)
print("EQUIPES FORMADAS:")
print("="*50)
print(df_teams)

# Análise detalhada
analyze_team_composition(df_teams)

# Visualizações
create_visualization(df_teams)

df_teams.to_csv('balanced_teams.csv', index=False)