# ECG Classification with CNN

Projeto de classificação de sinais de eletrocardiograma (ECG) utilizando redes neurais convolucionais 1D (CNN-1D) para detectar múltiplas condições cardíacas.

## 📋 Descrição

Este projeto implementa um modelo de deep learning para classificação multi-label de sinais de ECG de 12 derivações. O modelo é capaz de identificar as seguintes condições:

- **1dAVb**: Bloqueio atrioventricular de primeiro grau
- **RBBB**: Bloqueio de ramo direito
- **LBBB**: Bloqueio de ramo esquerdo
- **SB**: Bradicardia sinusal
- **AF**: Fibrilação atrial
- **ST**: Alteração do segmento ST
- **normal_ecg**: ECG normal

## 🏗️ Estrutura do Projeto

```
ecg/
├── ecgai/                         # Pacote principal
│   ├── __init__.py
│   ├── config.py                  # Sistema de configuração
│   ├── data/                      # Módulo de dados
│   │   ├── __init__.py
│   │   ├── dataset.py             # Dataset + DataLoaders
│   │   └── preprocess.py          # Pré-processamento
│   ├── models/                    # Módulo de modelos
│   │   ├── __init__.py
│   │   └── cnn.py                 # Arquitetura CNN-1D
│   ├── training/                  # Módulo de treinamento
│   │   ├── __init__.py
│   │   ├── trainer.py             # Classe Trainer
│   │   └── metrics.py             # Métricas
│   └── utils/                     # Módulo de utilitários
│       ├── __init__.py
│       ├── checkpoint.py          # Checkpoints
│       ├── distributed.py         # Funções DDP
│       ├── helpers.py             # Helpers gerais
│       └── logging.py             # Sistema de logs
├── notebooks/                     # Jupyter notebooks
│   ├── test_h5.ipynb             # Testes com HDF5
│   └── train.ipynb               # Notebook de treino
├── main.py                        # 👉 Ponto de entrada único
├── config.yaml                    # Configuração
├── pyproject.toml                 # Dependências + config (ruff, mypy, taskipy)
└── README.md                     # Este arquivo
```

## 🚀 Instalação

### 1. Instalar uv (package manager moderno)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Ou no macOS/Linux:
```bash
brew install uv
```

### 2. Clone o repositório

```bash
git clone <repository-url>
cd ecg
```

### 3. Instalar dependências

```bash
uv sync
```

Isso irá:
- Criar `.venv` automaticamente
- Instalar todas as dependências do `pyproject.toml`
- Gerar `uv.lock` para reprodutibilidade

### 4. Ativar ambiente (opcional)

```bash
source .venv/bin/activate
```

Ou use `uv run` para executar comandos diretamente sem ativar.

## 📊 Preparação dos Dados

O projeto espera dados em formato HDF5 com a seguinte estrutura:
- Diretório `dataset/unzipped/` com arquivos `exams_part*.hdf5`
- Arquivo `dataset/exams.csv` com metadados e labels

### Pré-processamento

Execute o pré-processamento:

```bash
uv run task preprocess
```

Ou com argumentos customizados:
```bash
uv run python main.py preprocess \
    --data-dir dataset/unzipped \
    --csv-path dataset/exams.csv \
    --output-dir processed_npz
```

Este script:
- Normaliza os sinais por derivação (z-score)
- Divide os dados em train/val/test por paciente (80/10/10)
- Salva arquivos `.npz` comprimidos em `processed_npz/`

## 🎯 Treinamento

### Configuração

Edite o arquivo `config.yaml` para ajustar hiperparâmetros:

```yaml
training:
  batch_size: 32
  epochs: 50
  learning_rate: 0.001
  patience: 7
```

### Executando o Treinamento

#### Com uv diretamente:
```bash
uv run python main.py train --config config.yaml
uv run torchrun --standalone --nproc_per_node=4 main.py train --config config.yaml
```

#### Com taskipy (recomendado):
```bash
uv run task train
uv run task train-ddp
uv run task preprocess
```

#### Ver tasks disponíveis:
```bash
uv run task --list
```

## 🛠️ Desenvolvimento

### Tasks Disponíveis (taskipy)

O projeto usa **taskipy** para automatizar tarefas comuns:

```bash
uv run task --list
```

#### Qualidade de Código
```bash
uv run task format       # Formata código com ruff
uv run task lint         # Verifica código com ruff
uv run task lint-fix     # Corrige problemas automaticamente
uv run task type-check   # Verifica tipos com mypy
uv run task check        # Roda lint + type-check
uv run task fix          # Corrige e formata tudo
```

#### Treinamento
```bash
uv run task train        # Treina modelo (single-process)
uv run task train-ddp    # Treina modelo (distribuído 4 GPUs/cores)
uv run task preprocess   # Pré-processa dados
```

#### Limpeza
```bash
uv run task clean        # Remove caches e arquivos temporários
uv run task clean-all    # Limpeza completa incluindo .venv
```

### ✨ Melhorias e Otimizações

#### Ferramentas Modernas
- **uv**: Package manager ultra-rápido
- **ruff**: Linting e formatting em Rust (10-100x mais rápido)
- **mypy**: Type checking estático
- **taskipy**: Gerenciamento de tasks simplificado

#### Arquitetura Modular
- **Configuração Externa**: YAML editável sem tocar no código
- **Classe Trainer**: Gerenciamento profissional do loop de treino
- **Type Hints**: Código totalmente tipado para melhor IDE support
- **Logging Estruturado**: Logs claros e informativos

#### Performance
- **I/O Otimizado**: Memory-mapped files (~30% mais rápido)
- **Auto-detect Workers**: Número ótimo de workers automaticamente
- **Gradient Clipping**: Previne explosão de gradientes
- **Mixed Precision**: Suporte a AMP para GPUs

#### Qualidade
- **Métricas Detalhadas**: F1, Precision, Recall, AUROC, AP
- **Checkpoints Inteligentes**: Salva best e last automaticamente
- **Early Stopping**: Para quando não há mais melhoria
- **Class Weights**: Balanceamento automático de classes

## 🧠 Arquitetura do Modelo

O modelo `ECG_CNN1D` consiste em:
- 3 blocos convolucionais (32 → 64 → 128 filtros)
- Batch Normalization e ReLU após cada convolução
- Global Average Pooling
- Camadas densas com Dropout (0.3)
- Saída com Sigmoid para classificação multi-label

**Input**: `(batch, 4096, 12)` - 4096 pontos temporais × 12 derivações  
**Output**: `(batch, 7)` - Probabilidades para cada condição

## 📈 Métricas

O modelo é avaliado usando:
- **F1-Score** (macro e micro)
- **Precision** (macro e micro)
- **Recall** (macro e micro)
- **AUROC** (macro e micro)
- **Average Precision** (PR-AUC)

Métricas são salvas automaticamente em:
- `checkpoints/training_metrics.csv` - Histórico completo
- `checkpoints/best_model.pt` - Melhor modelo
- `checkpoints/last_model.pt` - Último checkpoint

## 🔧 Requisitos

- Python 3.8+
- uv (package manager)
- PyTorch 2.0+
- CUDA (opcional, para GPU)
- 8GB+ RAM recomendado

### Instalação do uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 🔧 Configuração Avançada

### Ajustar Hiperparâmetros

Edite `config.yaml` para customizar:

```yaml
training:
  batch_size: 32          # Tamanho do batch
  learning_rate: 0.001    # Learning rate inicial
  patience: 7             # Early stopping patience
  eval_every: 5           # Avaliar a cada N épocas
  gradient_clip: 1.0      # Clipping de gradientes (null = desabilitado)
  warmup_epochs: 3        # Épocas de warmup
  use_amp: false          # Mixed precision (apenas GPU)
  compile_model: false    # torch.compile() (PyTorch 2.x)

dataloader:
  num_workers: null       # null = auto-detect
  pin_memory: false       # true para GPU
  prefetch_factor: 2

model:
  dropout_rate: 0.3       # Dropout rate
```

### Usar GPU

Se tiver GPU disponível, ajuste:

```yaml
training:
  use_amp: true          # Mixed precision training

dataloader:
  pin_memory: true       # Acelera transferência CPU->GPU

distributed:
  backend: "nccl"        # Backend otimizado para GPU
```

## 🧪 Testes e Validação

### Avaliar Modelo Treinado

```python
from ecgai import ECG_CNN1D, Config
from ecgai.utils import load_checkpoint
import torch

# Carrega modelo
config = Config.get_default()
model = ECG_CNN1D(
    n_leads=config.data.num_leads,
    n_classes=len(config.data.labels)
)
load_checkpoint('checkpoints/best_model.pt', model)
model.eval()

# Inferência
with torch.no_grad():
    signal = torch.randn(1, 12, 4096)  # [batch, leads, length]
    output = model(signal)  # Probabilidades [batch, 7]
    
print(f"Predições: {output}")
print(f"Classes: {config.data.labels}")
```

## 📝 Notas

- Os dados originais **não** estão incluídos no repositório
- Modelos treinados (`.pt`) são ignorados pelo git
- Logs e métricas são salvos automaticamente em `checkpoints/`
- Arquivos de backup (`.bak`) são ignorados pelo git

## 📄 Licença

[Adicione sua licença aqui]

## 👥 Autores

[Adicione os autores aqui]
