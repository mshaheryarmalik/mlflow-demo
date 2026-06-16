#!/usr/bin/env bash
# stop.sh — Stop the mlflow-demo stack
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'

log()  { echo -e "${GREEN}[mlflow-demo]${NC} $*"; }
warn() { echo -e "${YELLOW}[mlflow-demo]${NC} $*"; }

usage() {
  echo "Usage: ./stop.sh [--clean]"
  echo ""
  echo "  (no flags)   Stop containers, keep data volumes"
  echo "  --clean      Stop containers AND delete all volumes (resets all data)"
  echo ""
}

main() {
  local clean=false
  for arg in "$@"; do
    case "$arg" in
      --clean) clean=true ;;
      --help|-h) usage; exit 0 ;;
      *) echo "Unknown flag: $arg"; usage; exit 1 ;;
    esac
  done

  if ! docker info &>/dev/null; then
    warn "Docker daemon is not running — nothing to stop."
    exit 0
  fi

  if [[ "$clean" == true ]]; then
    warn "Stopping containers and removing all volumes (all MLflow data will be lost)..."
    docker compose -f "$ROOT/docker-compose.yml" down --volumes --remove-orphans
    log "Clean stop complete. All data volumes removed."
  else
    log "Stopping containers (data volumes preserved)..."
    docker compose -f "$ROOT/docker-compose.yml" down --remove-orphans
    log "Stack stopped. Run ./start.sh to restart."
  fi
}

main "$@"
