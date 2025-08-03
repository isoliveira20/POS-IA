**Tech Challenge - Fase 02**
Izabela de Souza Oliveira - RM 364554
Thais Tozatto - RM 
Rafael de Castro Almeida - RM 




**Geração de tasks**
Regras de negócio:
- Tipos de task distribuídos conforme perfil realista por stack (ex.: Backend tem mais bugs críticos, frontend mais tarefas planejadas).
- Prioridades variadas, com maior incidência em prioridades média e alta.
- Stacks com presença garantida de todas e distribuição baseada em pesos que refletem a demanda real.
- Complexidade com pontuações variadas de 1 a 8, com cobertura mínima para todas as complexidades.
- Horas estimadas baseadas na complexidade e ajustadas por multiplicadores específicos para cada stack, refletindo diferenças de esforço entre áreas.
- Pelo menos uma task de cada tipo, prioridade, stack e complexidade para assegurar diversidade no conjunto.

**Resultados**
min_total_hours: **2270**
Stack BE: mínimo estimado de horas = 1240
Stack DB: mínimo estimado de horas = 70
Stack FE: mínimo estimado de horas = 400
Stack MO: mínimo estimado de horas = 140
Stack DS: mínimo estimado de horas = 210
Stack DE: mínimo estimado de horas = 210

{'BE': 0, 'FE': 0, 'DE': 0, 'DS': 0, 'DB': 0, 'MO': 0}

Horas geradas: **2339**
Horas por Stack: **{'BE': 1247, 'FE': 418, 'DE': 234, 'DS': 223, 'DB': 72, 'MO': 145}**

*Quantidades por tipo de task:*
Bug               11
Critical Bug       8
Planned Task      58
Spike              7
Technical Debt     9
Unplanned Task     6

*Quantidade por prioridade de task:*
Critical    10
Disaster     4
High        33
Low         28
Medium      24

*Quantidade por stack de desenvolvimento:*
BE    50
DB     4
DE    12
DS     7
FE    21
MO     5

*Quantidade por complexidade de task:*
1     5
2     8
3    11
5    18
8    57
