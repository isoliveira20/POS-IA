## Tech Challenge - Fase 03
- Izabela de Souza Oliveira - RM 364554
- Thais Costa Tozatto - RM 363288
- Rafael Castro de Almeida - RM 362308

## Problema
Nesta fase, precisamos executar o fine-tuning de um foundation model, utilizando o dataset "The AmazonTitles-1.3MM". 
Como output, o modelo treinado deverá:
1. Receber perguntas com um contexto obtido por meio do arquivo json “trn.json” contido no dataset.
2. A partir do prompt sobre o título do produto, o nosso modelo escolhido deverá gerar uma resposta baseada na pergunta do usuário trazendo como resultado do aprendizado do fine-tuning os dados da sua descrição.

O nosso grupo escolheu fazer o **Fine-tuning com Unsloth (Llama-3.2-3B-Instruct 4-bit)**.

Inicialmente nós tentamos outras abordagens como o bert-base-uncased e o google/flan-t5-small. Com o bert não tivemos sucesso devido as limitações de máquinas e tamanho do modelo. Com isso, fizemos testes utilizando o flan, porém o loss ficou sempre acima de 3.5. Também tentamos realizar o treinamento de duas formas:
 - Maquinas com GPU
 - Em memória

O resultado foi similar, não conseguimos deixar abaixo de 3.5.

Devido a isso, seguimos para outros modelos, obtendo bom resultado no Llama.

---

## 🔎 Visão Geral

O pipeline implementa os seguintes passos:

1. **Instalação** – configuração do ambiente e dependências (Unsloth, Transformers, TRL, PEFT).
2. **Preparação de dados** – limpeza e formatação de prompts/respostas a partir de dataset público no Hugging Face.
3. **Modelo base** – carregamento de `Llama-3.2-3B-Instruct` em 4-bit (QLoRA-ready).
4. **Treinamento LoRA** – aplicação de Low-Rank Adaptation em camadas de atenção (Q/V).
5. **SFT Trainer** – treino supervisionado com Hugging Face TRL.
6. **Avaliação rápida** – testes de memorização e generalização via prompts.
7. **Persistência** – checkpoints intermediários e modelo final com adaptadores PEFT.
8. **Inferência** – comparação entre modelo base e modelo treinado.

---

## 🖥️ Pré-requisitos

- Acesso a GPU (Google Colab ou servidor com CUDA).  
- 12–16 GB de VRAM são suficientes para rodar em 4-bit.  
- Conta no [Hugging Face Hub](https://huggingface.co/) para carregar datasets.  
- (Opcional) Conta no [Weights & Biases](https://wandb.ai/) para log de métricas.  

---

## ⚙️ Instalação do Ambiente

```bash
!pip uninstall -y transformers trl unsloth unsloth_zoo
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps torch xformers "trl<0.9.0" peft accelerate bitsandbytes triton
!pip install datasets==4.0.0
```
**Observações**

* --no-deps evita conflitos de versão.
* trl<0.9.0 garante compatibilidade com SFTTrainer.
* datasets==4.0.0 é antigo; se houver erros, usar datasets>=2.20.0.

---
## 📑 Configurações

Nosso treinamento foi realizado através de checkpoints, permitindo melhor controle de recursos durante a execução. Em nossas configurações, foram criadas três opções de execution mode:
 - INFERENCE_ONLY -> Utiliza o modelo já treinado
 - TRAIN_FULL -> Inicia o treinamento completo do modelo
 - TRAIN_RESUME -> Continua o treinamento à partir do checkpoint

```bash
MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
MAX_LENGTH = 512
BATCH_SIZE = 2
EPOCHS = 2
LEARNING_RATE = 5e-5
DATASET_SIZE = 50000

RESUME_CHECKPOINT_PATH = "/content/drive/MyDrive/unsloth_checkpoints/checkpoint-13000"
FINAL_MODEL_PATH = "/content/drive/MyDrive/unsloth_model"
CHECKPOINT_SAVE_DIR = "/content/drive/MyDrive/unsloth_checkpoints"

EXECUTION_MODE = "INFERENCE_ONLY"  # ou TRAIN_FULL | TRAIN_RESUME
```
---
## 📊 Preparação dos Dados

Para acesso aos dados, nós inicialmente tentamos utilizar o drive, mas por performance optamos pela transferência para o Hugging Face.

1. Carregar dataset Hugging Face:
```bash
dataset = load_dataset("thaistozatto/techchalleng03_trn")
```
2. Selecionar colunas title e content.
3. Remover valores nulos ou vazios.
4. Criar pares prompt / response no formato de diálogo:
Exemplo de prompt:
```bash
User: Provide a detailed description of the product titled '<title>'.
Assistant:
```
---

## 🤖 Modelo e Treinamento

* Carregamento do modelo base em 4-bit com FastLanguageModel.
* Preparação com LoRA para treino eficiente em GPU limitada.
* Configuração via TrainingArguments com avaliação e checkpoints a cada 500 steps.
* Treino supervisionado usando SFTTrainer da TRL.
---

## Modos de execução
* TRAIN_FULL: Treino do zero + testes antes/depois.
* TRAIN_RESUME: Retoma de checkpoint.
* INFERENCE_ONLY: Carrega modelo salvo e roda apenas inferência.
---

## 🧪 Avaliação
O script inclui uma função test_model() que avalia:
* **Memorização:** prompts já vistos no treino.
* **Generalização:** prompts novos, não vistos no dataset.

Exemplo de uso:
```bash
test_model(model, tokenizer, [
    "Provide a detailed description of the product titled 'Instant Pot Duo Plus 9-in-1 Electric Pressure Cooker'",
    "Provide a detailed description of the product titled 'The Hobbit: An Unexpected Journey'"
])
```
---
## 📂 Saídas
* Checkpoints: salvos em CHECKPOINT_SAVE_DIR.
* Modelo final: adaptadores LoRA + tokenizer em FINAL_MODEL_PATH.
---

## 📊 Métricas de Treinamento

**1. Train Loss**
* Começou em torno de 2.5 e caiu rapidamente até **~2.2**.
* Depois, estabilizou com pequenas oscilações até o final **(~2.15–2.2)**.
→ Indica que o modelo aprendeu nas primeiras iterações e depois entrou em um plateau estável.

**2. Learning Rate**
* Curva decrescente com “resets” típicos de scheduler (provavelmente cosine ou linear decay with warmup).
* O ajuste foi consistente: começou alto (~5e-5) e caiu gradualmente até próximo de zero.

**3. Gradient Norm**
* Oscilou entre **0.8 e 1.5**, com alguns picos **(~2.5)**.
* Esses picos são normais em lotes difíceis, mas como não são recorrentes, não indicam instabilidade séria.

**4. Global Step & Epoch**
* Subida linear, sem interrupções: *o treino rodou até o final de 2 épocas e ~12.500 steps.*

![metricas_](https://github.com/user-attachments/assets/a8ff15e1-b251-411f-aab0-57531ab11e89)
---

## 🔹 Métricas de Validação
**1. Eval Loss**
* Caiu de **~2.26 para ~2.18**, de forma consistente.
* A curva tem formato “descendente suave” → o modelo generalizou, não apenas decorou.

**2. Velocidade (steps/s, samples/s)**
* Estável em torno de 1.39 steps/s e ~11.1 samples/s.
* Pequenas variações são esperadas, sem degradação com o tempo.

**3. Runtime**
* Ficou constante em ~261 segundos por avaliação.
* Picos em 10k steps, mas logo voltou ao normal.
---

**Conclusão da Análise**
* Convergência: O modelo aprendeu de forma estável. A queda de loss é clara no treino e na avaliação.
* Generalização: Como o eval loss acompanhou o train loss (sem aumentar), não houve sinal forte de overfitting.
* Estabilidade: Grad norm controlado, sem explosões que indicassem instabilidade numérica.
* Eficiência: Taxa de processamento consistente, indicando bom aproveitamento de GPU.

**👉 Em resumo: o fine-tuning foi bem-sucedido. O modelo convergiu, manteve estabilidade e apresenta indícios de boa capacidade de generalização.**
![metricas](https://github.com/user-attachments/assets/54149f31-f44f-453d-bdb3-3ea3b2af80e5)



