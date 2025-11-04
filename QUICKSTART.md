# ECGAI - Quick Start

Guia rápido para começar a usar o projeto.

## Instalação Rápida

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone <repository-url>
cd ecg

uv sync
```

## Comandos Principais

### Ver todas as tasks disponíveis
```bash
uv run task --list
```

### Treinamento
```bash
uv run task train
uv run task train-ddp
```

### Qualidade de Código
```bash
uv run task format
uv run task lint
uv run task type-check
uv run task fix
```

### Pré-processamento
```bash
uv run task preprocess
```

### Limpeza
```bash
uv run task clean
uv run task clean-all
```

## Estrutura de Tasks (pyproject.toml)

Todas as tasks estão configuradas em `pyproject.toml`:

```toml
[tool.taskipy.tasks]
format = "ruff format ."
lint = "ruff check ."
lint-fix = "ruff check --fix ."
type-check = "mypy ecgai main.py"
check = "task lint && task type-check"
fix = "task lint-fix && task format"

train = "python main.py train --config config.yaml"
train-ddp = "torchrun --standalone --nproc_per_node=4 main.py train --config config.yaml"
preprocess = "python main.py preprocess --data-dir dataset/unzipped --csv-path dataset/exams.csv"

clean = "rm -rf __pycache__ **/__pycache__ .mypy_cache .ruff_cache checkpoints/*.pt *.log"
clean-all = "task clean && rm -rf .venv uv.lock"
```

## Ferramentas Configuradas

- **uv**: Package manager ultra-rápido (substitui pip)
- **ruff**: Linter + formatter em Rust (substitui black, flake8, isort)
- **mypy**: Type checker estático
- **taskipy**: Gerenciador de tasks (substitui make)

## Próximos Passos

1. Configure seus dados em `dataset/`
2. Ajuste hiperparâmetros em `config.yaml`
3. Execute: `uv run task preprocess`
4. Execute: `uv run task train`
5. Monitore métricas em `checkpoints/training_metrics.csv`

## Workflow de Desenvolvimento

```bash
uv run task format
uv run task lint-fix
uv run task type-check

uv run task train

uv run task clean
```

Pronto para uso!
