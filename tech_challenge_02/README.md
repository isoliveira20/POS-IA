## Tech Challenge - Fase 02
- Izabela de Souza Oliveira - RM 364554
- Thais Costa Tozatto - RM
- Rafael Castro de Almeida - RM 

## Problema
O problema proposto considera a utilização de **Algoritmos Genéticos (AG)** para otimizar a alocação de tarefas de desenvolvimento para equipes em sprints de 10 dias. O sistema considera múltiplas variáveis como compatibilidade de stack tecnológica, níveis de senioridade, prioridades das tarefas e balanceamento de carga de trabalho.

## Objetivo geral 
Encontrar a melhor distribuição de tarefas entre desenvolvedores, minimizando:
- Sobrecarga de trabalho
- Custos operacionais
- Incompatibilidades técnicas
- Desequilíbrio na distribuição de tarefas

## Estrutura dos Dados
### Datasets Utilizados
- **`teams_dataset.csv`**: Informações dos desenvolvedores (nome, nível, stack)
- **`tasks_dataset.csv`**: Detalhes das tarefas (stack, prioridade, horas estimadas)

### Níveis de Desenvolvedores
| Nível | Multiplicador de Custo | Horas Máx/Dia | Tolerância Sobrecarga | Penalidade |
|-------|------------------------|----------------|----------------------|------------|
| Intern | 0.5x | 5h | 0h | 1000x |
| Junior | 0.75x | 7h | 2h | 500x |
| Mid-level | 1.0x | 7h | 5h | 200x |
| Senior | 1.25x | 7h | 10h | 100x |
| Specialist | 1.5x | 7h | 10h | 100x |

### Prioridades das Tarefas
- **Disaster**: Peso 5 (máxima prioridade)
- **Critical**: Peso 4
- **High**: Peso 3
- **Medium**: Peso 2
- **Low**: Peso 1 (menor prioridade)

## Algoritmo Genético
### Representação do Cromossomo
Cada solução é representada como uma lista onde cada posição corresponde a uma tarefa e o valor indica qual desenvolvedor foi alocado:
```
[dev1, dev3, dev2, dev1, dev4, ...]
 task1 task2 task3 task4 task5
```

### Geração da População Inicial
Três estratégias combinadas para criar diversidade:

1. **Prioritizada**: Aloca tarefas por ordem de prioridade
2. **Balanceada**: Distribui baseada na carga atual dos desenvolvedores
3. **Experiente**: Prioriza devs Senior/Specialist para tarefas críticas

### Função de Fitness
A função objetivo combina múltiplos critérios com pesos específicos:

```python
fitness = w_overload × sobrecarga + w_cost × custo_total + 
          w_imbalance × desequilíbrio + w_distribution × distribuição + 
          w_penalties × penalidades
```

**Pesos utilizados:**
- Sobrecarga: 0.6
- Custo: 0.08
- Desequilíbrio: 0.4
- Distribuição: 0.15
- Penalidades base: 0.1

### Operadores Genéticos

#### Seleção
- **Torneio**: Definição dos pais por torneio. Seleciona o melhor entre k=7 indivíduos aleatórios

#### Crossover
- **Two-point crossover**: Troca segmentos entre dois pontos
- **Uniform crossover**: Escolhe aleatoriamente genes de cada pai

#### Mutação Adaptativa
- Taxa inicial: 5%
- Taxa máxima: 50%
- Aumenta automaticamente após 5 gerações sem melhoria
- Estratégia balanceada: prioriza devs com menor carga

### Busca Local
Aplicada apenas aos 3 melhores indivíduos (elite):
- **Swaps bidirecionais**: Troca tarefas entre devs compatíveis
- **Busca incremental**: Otimização iterativa com controle de recursos

### Elitismo e Diversidade
- **Elite**: Top 3 soluções preservadas a cada geração
- **Injeção de diversidade**: A cada 30 gerações, 10% da população é renovada
- **Correção de incompatibilidades**: Garante que todos os genes sejam válidos

## Parâmetros de Configuração

```python
num_generations = 300           # Número máximo de gerações
population_size = len(tasks) × 2  # Tamanho da população
sprint_days = 10               # Duração do sprint
max_no_improve = 20            # Critério de parada por estagnação
tournament_size = 7            # Tamanho do torneio
elite_size = 3                 # Número de indivíduos preservados
```

## Métricas e Resultados

### Resultados
- População inicial gerada com **198** soluções
- Fitness:
  - Melhor fitness: **797.46**
  - Fitness médio: **856.84**
  - Geração **184**

### Critérios de Avaliação
1. **Compatibilidade**: Todas as tarefas alocadas a devs com stack compatível
2. **Balanceamento**: Distribuição equilibrada de horas de trabalho
3. **Respeito aos limites**: Nenhum dev excede significativamente sua capacidade
4. **Priorização**: Tarefas críticas recebem atenção adequada

### Visualizações Geradas
- Evolução do fitness ao longo das gerações
- Distribuição de horas por desenvolvedor
- Detalhamento de tarefas por dev (stack, prioridade, horas)

## Conclusões
A utilização de AG como meio para solução do problema de alocação de tarefas por sprint trouxe algumas vantagens:
- **Otimização multi-objetivo**: Considerando múltiplos fatores simultaneamente
- **Adaptabilidade**: Mutação adaptativa previnindo convergência prematura
- **Robustez**: Correção automática de incompatibilidades
- **Eficiência**: Busca local acelerou convergência
- **Flexibilidade**: Facilmente customizável para diferentes cenários
Todos essas vantagens nos permitiram ajustar e customizar os parâmetros conforme os testes foram realizados, até chegarmos em uma solução realista e satisfatório.

-----------------
## Geração de equipes
- Regras de negócio:
  - Experiência selecionada por peso/probabilidade.
  - Máximo 2 especialistas por equipe, no máximo 1 especialista por stack.
  - Presença obrigatória de pelo menos um Senior ou Specialist com stack Backend (BE) por equipe.
  - Máximo 1 Intern e 1 Junior por equipe.
  - Máximo de Seniors por equipe definido por tamanho: 1 (até 4 membros), 2 (5 a 6 membros), 3 (acima de 6 membros).
  - Maioria Backend no time.
  - Diversidade mínima de stacks ajustada ao tamanho da equipe (mínimo 2 para equipes pequenas, mínimo 3 para médias e maior para times grandes).

### Resultados
- Formação de 7 equipes multidisciplinares
- Total de desenvolvedores: **33**
- BE Global: 18/33 (54.5%)

- *Distribuição por nível:*
  - Mid-level: 13 (39.4%)
  - Senior: 11 (33.3%)
  - Junior: 4 (12.1%)
  - Specialist: 3 (9.1%)
  - Intern: 2 (6.1%)

- *Distribuição por stack:*
  - BE: 18 (54.5%)
  - FE: 6 (18.2%)
  - DE: 3 (9.1%)
  - DS: 3 (9.1%)
  - MO: 2 (6.1%)
  - DB: 1 (3.0%)

- *Distribuição por experiência média por equipe:*
    - A: 4.4
    - B: 3.9
    - C: 4.3
    - D: 4.4
    - E: 6.0
    - F: 6.0
--------------------
## Geração de tasks
- Regras de negócio:
  - Tipos de task distribuídos conforme perfil realista por stack (ex.: Backend tem mais bugs críticos, frontend mais tarefas planejadas).
  - Prioridades variadas, com maior incidência em prioridades média e alta.
  - Stacks com presença garantida de todas e distribuição baseada em pesos que refletem a demanda real.
  - Complexidade com pontuações variadas de 1 a 8, com cobertura mínima para todas as complexidades.
  - Horas estimadas baseadas na complexidade e ajustadas por multiplicadores específicos para cada stack, refletindo diferenças de esforço entre áreas.
  - Pelo menos uma task de cada tipo, prioridade, stack e complexidade para assegurar diversidade no conjunto.

### Resultados
- min_total_hours: **2270**
- Stack BE: mínimo estimado de horas = 1240
- Stack DB: mínimo estimado de horas = 70
- Stack FE: mínimo estimado de horas = 400
- Stack MO: mínimo estimado de horas = 140
- Stack DS: mínimo estimado de horas = 210
- Stack DE: mínimo estimado de horas = 210

- {'BE': 0, 'FE': 0, 'DE': 0, 'DS': 0, 'DB': 0, 'MO': 0}

- Horas geradas: **2339**
- Horas por Stack: **{'BE': 1247, 'FE': 418, 'DE': 234, 'DS': 223, 'DB': 72, 'MO': 145}**

- *Quantidades por tipo de task:*
    - Bug: 11
    - Critical Bug: 8
    - Planned Task: 58
    - Spike: 7
    - Technical Debt: 9
    - Unplanned Task: 6

- *Quantidade por prioridade de task:*
  - Critical: 10
  - Disaster: 4
  - High: 33
  - Low: 28
  - Medium: 24

- *Quantidade por stack de desenvolvimento:*
  - BE: 50
  - DB: 4
  - DE: 12
  - DS: 7
  - FE: 21
  - MO: 5
 
- *Quantidade por complexidade de task:*
  - 1 | 5
  - 2 | 8
  - 3 | 11
  - 5 | 18
  - 8 | 57
