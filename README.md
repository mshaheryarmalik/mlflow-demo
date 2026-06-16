# mlflow-demo

> **A production-level MLflow learning project** covering both GenAI (LangGraph + Ollama) and Classical ML (PyTorch), with full observability, evaluation, and model registry — all running locally via Docker.

---

## Project Layout

```
mlflow-demo/
├── docker-compose.yml       # MLflow + PostgreSQL + MinIO + Ollama
├── env.example              # Copy to .env
│
├── llms-agents/             # LangGraph multi-agent pipeline with MLflow tracing
│   ├── pyproject.toml
│   ├── src/agents/
│   │   ├── state.py         # Shared AgentState (TypedDict)
│   │   ├── tools.py         # Custom tools (web search stub, calculator)
│   │   ├── nodes.py         # Researcher / Writer / Critic nodes via Ollama
│   │   └── graph.py         # Compiled LangGraph StateGraph
│   ├── scripts/
│   │   └── run_pipeline.py  # End-to-end pipeline runner
│   ├── notebooks/
│   │   ├── 01_tracing_demo.ipynb
│   │   ├── 02_evaluation.ipynb
│   │   └── 03_prompt_registry.ipynb
│   └── tests/
│       └── test_agents.py
│
└── model-training/          # PyTorch CNN on MNIST with MLflow tracking
    ├── pyproject.toml
    ├── src/training/
    │   ├── config.py        # Dataclass-based hyperparameter config
    │   ├── dataset.py       # PyTorch Dataset + DataLoader helpers
    │   ├── model.py         # CNN architecture
    │   ├── train.py         # Training loop (mlflow.pytorch.autolog)
    │   └── evaluate.py      # Evaluation + model registry promotion
    ├── scripts/
    │   └── run_training.py  # Hydra-style CLI entry point
    ├── notebooks/
    │   ├── 01_experiment_tracking.ipynb
    │   └── 02_model_registry.ipynb
    └── tests/
        └── test_training.py
```

---

## MLflow Features Demonstrated

| Feature | Where |
|---|---|
| **Tracing / Autolog** | `llms-agents` — every LangGraph step traced |
| **Prompt Registry** | `llms-agents` — versioned system prompts |
| **LLM Evaluation** | `llms-agents` — relevance, faithfulness metrics |
| **Experiment Tracking** | `model-training` — loss, accuracy per epoch |
| **PyTorch Autolog** | `model-training` — automatic metric/param capture |
| **Model Registry** | `model-training` — staging → production promotion |
| **Model Serving** | `model-training` — `mlflow models serve` |

---

## Infrastructure

| Service | URL | Purpose |
|---|---|---|
| MLflow UI | http://localhost:5000 | Experiments, runs, registry |
| MinIO Console | http://localhost:9001 | Artifact browser (user: `minioadmin`) |
| Ollama | http://localhost:11434 | Local LLM inference |
| PostgreSQL | localhost:5432 | MLflow backend store |

---

## Quick Start

### 1. Prerequisites
```bash
# macOS
brew install uv docker
```

### 2. Start Infrastructure
```bash
cp env.example .env
docker compose up -d
# Wait ~60 s for Ollama to pull the model
docker compose logs -f ollama
```

### 3. LLMs & Agents
```bash
cd llms-agents
uv sync
uv run python scripts/run_pipeline.py --topic "MLflow in production"
# Open http://localhost:5000 → Experiments → llm-agents-demo → Traces
```

### 4. Model Training
```bash
cd model-training
uv sync
uv run python scripts/run_training.py --epochs 5 --lr 1e-3
# Open http://localhost:5000 → Experiments → mnist-cnn
```

---

## Tech Stack

- **Python 3.12** + **uv** (package management)
- **MLflow 2.x** (tracking, registry, evaluation, tracing)
- **LangGraph** (multi-agent orchestration)
- **LangChain + Ollama** (LLM integration)
- **PyTorch** (CNN training)
- **PostgreSQL 16** (MLflow backend)
- **MinIO** (artifact store, S3-compatible)
- **Docker Compose** (full local stack)
