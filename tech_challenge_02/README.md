***Tech Challenge - Fase 02***
- Izabela de Souza Oliveira - RM 364554
- Thais Costa Tozatto - RM
- Rafael Castro de Almeida - RM 

**Problema:**
- O problema proposto considera a utilização de Algoritmos Genéticos para otimizar a alocação de recursos de desenvolvimento de software, considerando os recursos humanos, horas por sprint, complexidade das tasks, priorização e senioridade do desenvolvedor.

**Algoritmo Genético**
- Regras de negócio:
  - *Escolha de pais por torneio*
  - *População inicial diversificada:* geramos soluções iniciais usando três estratégias diferentes, incluindo priorizar tarefas críticas para devs mais experientes. O objetivo é ampliar a diversidade e melhorar a qualidade das soluções desde o início.
  - *Função fitness aprimorada:* penalidades melhor ajustadas para sobrecarga, incompatibilidade e balanceamento entre níveis.
  Reescalonamento dos pesos para resultados mais estáveis e na faixa adequada (centenas).
  - *Busca local otimizada:* aplicamos somente nos 3 melhores indivíduos da geração para refinar soluções. Utilizamos a atualização incremental para não deixar lento, melhorando o balanceamento interno.
  - *Mutação adaptativa e seleção mais seletiva:* taxa de mutação varia conforme a evolução, ajudando a escapar de mínimos locais. Torneio de seleção maior (k=7) para favorecer melhores soluções.
  - *Injeção periódica de diversidade:* a cada 30 gerações, adiciona novos indivíduos aleatórios para evitar estagnação.
  - *Correção inteligente de incompatibilidades:* corrigimos alocações incompatíveis priorizando devs menos sobrecarregados.

**Resultados:**
- População inicial gerada com *198* soluções
- Fitness:
  - Melhor fitness: **797.46**
  - Fitness médio: **856.84**
  - Geração **184**
  - *Parando por estagnação!*

**Geração de equipes**
- Regras de negócio:
  - Experiência selecionada por peso/probabilidade.
  - Máximo 2 especialistas por equipe, no máximo 1 especialista por stack.
  - Presença obrigatória de pelo menos um Senior ou Specialist com stack Backend (BE) por equipe.
  - Máximo 1 Intern e 1 Junior por equipe.
  - Máximo de Seniors por equipe definido por tamanho: 1 (até 4 membros), 2 (5 a 6 membros), 3 (acima de 6 membros).
  - Maioria Backend no time.
  - Diversidade mínima de stacks ajustada ao tamanho da equipe (mínimo 2 para equipes pequenas, mínimo 3 para médias e maior para times grandes).

**Resultados**
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

**Geração de tasks**
- Regras de negócio:
  - Tipos de task distribuídos conforme perfil realista por stack (ex.: Backend tem mais bugs críticos, frontend mais tarefas planejadas).
  - Prioridades variadas, com maior incidência em prioridades média e alta.
  - Stacks com presença garantida de todas e distribuição baseada em pesos que refletem a demanda real.
  - Complexidade com pontuações variadas de 1 a 8, com cobertura mínima para todas as complexidades.
  - Horas estimadas baseadas na complexidade e ajustadas por multiplicadores específicos para cada stack, refletindo diferenças de esforço entre áreas.
  - Pelo menos uma task de cada tipo, prioridade, stack e complexidade para assegurar diversidade no conjunto.

**Resultados**
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
