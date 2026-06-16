#!/usr/bin/env bash
# start.sh — Start the full mlflow-demo stack
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Colours ───────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'

log()  { echo -e "${GREEN}[mlflow-demo]${NC} $*"; }
info() { echo -e "${CYAN}[mlflow-demo]${NC} $*"; }
warn() { echo -e "${YELLOW}[mlflow-demo]${NC} $*"; }

# ── Prerequisites check ───────────────────────────────────────────────────────
check_deps() {
  for cmd in docker uv; do
    if ! command -v "$cmd" &>/dev/null; then
      echo "ERROR: '$cmd' is not installed." >&2
      [[ "$cmd" == "docker" ]] && echo "  Install: https://docs.docker.com/get-docker/" >&2
      [[ "$cmd" == "uv" ]]     && echo "  Install: curl -Ls https://astral.sh/uv/install.sh | sh" >&2
      exit 1
    fi
  done
  if ! docker info &>/dev/null; then
    echo "ERROR: Docker daemon is not running. Start Docker Desktop first." >&2
    exit 1
  fi
}

# ── .env setup ────────────────────────────────────────────────────────────────
setup_env() {
  if [[ ! -f "$ROOT/.env" ]]; then
    warn ".env not found — copying from env.example"
    cp "$ROOT/env.example" "$ROOT/.env"
    log ".env created. Edit it if you need custom credentials."
  fi
}

# ── Docker stack ──────────────────────────────────────────────────────────────
start_docker() {
  log "Starting Docker services (postgres, minio, mlflow, ollama)..."
  docker compose -f "$ROOT/docker-compose.yml" up -d

  log "Waiting for MLflow server to be ready..."
  local retries=30
  until curl -sf http://localhost:5000/health &>/dev/null; do
    retries=$((retries - 1))
    if [[ $retries -le 0 ]]; then
      echo "ERROR: MLflow server did not start in time." >&2
      docker compose -f "$ROOT/docker-compose.yml" logs mlflow | tail -20
      exit 1
    fi
    sleep 3
  done
  log "MLflow server is up."
}

# ── Python environments ───────────────────────────────────────────────────────
sync_envs() {
  for pkg in llms-agents model-training; do
    if [[ -f "$ROOT/$pkg/pyproject.toml" ]]; then
      log "Syncing Python environment for $pkg..."
      (cd "$ROOT/$pkg" && uv sync --quiet)
    fi
  done
}

# ── Main ──────────────────────────────────────────────────────────────────────
main() {
  echo ""
  info "============================================"
  info "  mlflow-demo  —  starting up"
  info "============================================"
  echo ""

  check_deps
  setup_env
  start_docker
  sync_envs

  echo ""
  log "Stack is ready!"
  echo ""
  echo -e "  ${CYAN}MLflow UI${NC}      →  http://localhost:5000"
  echo -e "  ${CYAN}MinIO Console${NC}  →  http://localhost:9091  (minioadmin / minioadmin)"
  echo -e "  ${CYAN}Ollama API${NC}     →  http://localhost:11434"
  echo ""
  echo -e "  ${GREEN}Run the LangGraph pipeline:${NC}"
  echo "    cd llms-agents && uv run python scripts/run_pipeline.py"
  echo ""
  echo -e "  ${GREEN}Run model training:${NC}"
  echo "    cd model-training && uv run python scripts/run_training.py"
  echo ""
  echo -e "  ${YELLOW}To stop:${NC}  ./stop.sh"
  echo ""
}

main "$@"
