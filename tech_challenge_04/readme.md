## Tech Challenge - Fase 04 ##
- Izabela de Souza Oliveira - RM 364554
- Thais Costa Tozatto - RM 363288
- Rafael Castro de Almeida - RM 362308

## Problema

Criação de uma aplicação que utilize análise de vídeo. O nosso projeto incorpora as técnicas de reconhecimento facial, análise de expressões emocionais em vídeos e detecção de atividades.


## 🚀 Features

👤 Detecção e tracking de pessoas

🧑‍🦱 Reconhecimento facial (rostos conhecidos)

🙂 Análise de emoções faciais

🏃‍♂️ Classificação de atividades corporais

🤝 Detecção de aperto de mão:

Plano aberto (pose corporal)

Close-up (detecção de mãos)

🎨 Geração de vídeo anotado

📊 Relatório final agregado

🧠 Modelos Utilizados

InsightFace → detecção facial + embeddings

DeepFace → análise de emoção

YOLOv8 Pose → keypoints corporais e atividades

MediaPipe Hand Landmarker → detecção precisa de mãos

## 📁 Estrutura Esperada 
tech_04/

├── video/

│   └── input_video.mp4

├── known_faces/

│   ├── pessoa1.jpg

│   └── pessoa2.png

├── output_analysis.mp4


video/ → vídeo de entrada

known_faces/ → imagens de pessoas conhecidas

output_analysis.mp4 → vídeo final gerado

## ⚙️ Dependências

Principais bibliotecas usadas:

- opencv-python
- numpy
- scipy
- insightface
- deepface
- ultralytics
- mediapipe
- tensorflow


O script já inclui comandos de instalação e configuração para execução no Google Colab com GPU.

## ▶️ Como Executar

1. Abra o projeto no Google Colab
2. Monte o Google Drive
3. Garanta que:
   - O vídeo de entrada existe
   - A pasta known_faces contém imagens válidas
   - Execute o script/notebook completo
   - Aguarde o processamento
   - O vídeo anotado será salvo automaticamente no Drive.

## 🔄 Pipeline (Resumo)

Para cada frame do vídeo:

- Detecção corporal (YOLO Pose)
- Detecção e reconhecimento facial
- Tracking de pessoas por centróide do rosto
- Associação rosto ↔ corpo
- Compensação de movimento da câmera
- Análise de atividade corporal
- Análise de emoção facial
- Detecção de aperto de mão (YOLO + MediaPipe)
- Desenho de overlays e labels
- Atualização de métricas globais


## 🎨 Convenções Visuais

🟢 Verde → estado normal

🔴 Vermelho → anomalia de movimento

🟣 Roxo → aperto de mão

Skeleton corporal desenhado sobre o corpo

Label:

**NOME | EMOÇÃO | ATIVIDADE**

## 📊 Relatório Final

Ao final do processamento, o script imprime:
- Total de frames processados
- Número de anomalias
- Ranking de atividades detectadas
<img width="2456" height="248" alt="image" src="https://github.com/user-attachments/assets/6c1afb70-6cbc-4f1d-9622-3f7600912cbb" />

## ⚠️ Limitações

- Emoção depende de boa visibilidade do rosto
- Reconhecimento facial exige imagens prévias de qualidade
- Tracking é baseado em proximidade espacial (não é Re-ID persistente)
- Oclusões podem afetar detecção de handshake

