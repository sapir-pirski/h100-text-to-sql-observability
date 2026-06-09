#!/usr/bin/env bash
#
# End-to-end runner for the MLOps assignment.
#
# This captures the workflow used on the Nebius H100 VM:
#   1. install/sync Python deps with uv
#   2. create .env if missing
#   3. load the BIRD subset
#   4. start Prometheus/Grafana/Langfuse with Docker Compose
#   5. optionally start vLLM with the H100 config
#   6. start the agent with the task-required verify/revise loop
#   7. optionally run evals and load tests
#
# Examples:
#   ./scripts/run-full-project.sh setup
#   ./scripts/run-full-project.sh stack
#   ./scripts/run-full-project.sh agent
#   ./scripts/run-full-project.sh h100-final
#   ./scripts/run-full-project.sh health
#   ./scripts/run-full-project.sh stop-all
#   ./scripts/run-full-project.sh eval
#   ./scripts/run-full-project.sh load-full
#   ./scripts/run-full-project.sh package
#
# Notes:
#   - This script does not write secrets. Put HF_TOKEN and Langfuse keys in
#     .env yourself.
#   - Starting vLLM and load-full consume H100 time.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

UV_BIN="${UV_BIN:-uv}"
DOCKER_BIN="${DOCKER_BIN:-docker}"
VLLM_MODEL_FINAL="${VLLM_MODEL_FINAL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
AGENT_HOST="${AGENT_HOST:-0.0.0.0}"
AGENT_PORT="${AGENT_PORT:-8001}"
AGENT_WORKERS="${AGENT_WORKERS:-16}"
LOAD_RPS="${LOAD_RPS:-10}"
LOAD_DURATION="${LOAD_DURATION:-300}"
LOAD_WARMUP_UNIQUE="${LOAD_WARMUP_UNIQUE:-0}"
LOAD_WARMUP_CONCURRENCY="${LOAD_WARMUP_CONCURRENCY:-16}"
LOAD_REQUEST_TIMEOUT_SECONDS="${LOAD_REQUEST_TIMEOUT_SECONDS:-120}"
LOAD_WARMUP_TIMEOUT_SECONDS="${LOAD_WARMUP_TIMEOUT_SECONDS:-$LOAD_REQUEST_TIMEOUT_SECONDS}"
LOAD_WARMUP_RETRIES="${LOAD_WARMUP_RETRIES:-0}"
LOG_DIR="$ROOT/logs"
RESULTS_DIR="$ROOT/results"
SUBMISSION_DIR="${SUBMISSION_DIR:-$ROOT/submission}"
SUBMISSION_ZIP="${SUBMISSION_ZIP:-$SUBMISSION_DIR/mlops-assignment-submission.zip}"
AGENT_PID_FILE="$LOG_DIR/agent.pid"
VLLM_PID_FILE="$LOG_DIR/vllm.pid"
CONFIG_FILE="${CONFIG_FILE:-}"
SUBMISSION_MANIFEST="${SUBMISSION_MANIFEST:-$ROOT/config/submission-manifest.txt}"

mkdir -p "$LOG_DIR" "$RESULTS_DIR"

log() {
    printf '[run-full-project] %s\n' "$*"
}

usage() {
    cat <<'EOF'
End-to-end runner for the MLOps assignment.

Actions:
  default       setup + stack + agent + health
  setup         uv sync --frozen, create .env if missing, load BIRD data
  stack         start Prometheus, Grafana, Langfuse with Docker Compose
  start-observability
                alias for stack
  stop-observability
                stop Prometheus, Grafana, Langfuse, and backing services
  vllm          start vLLM with scripts/start_vllm.sh
  start-vllm    alias for vllm
  stop-vllm     stop the vLLM process
  agent         start the FastAPI agent with the verify/revise loop enabled
  start-agent   alias for agent
  stop-agent    stop the agent process
  stop-all      stop agent, vLLM, and observability services
  health        check agent, vLLM, Prometheus, and Grafana health
  eval          run baseline eval into results/eval_baseline.json
  eval-after    run post-tuning eval into results/eval_after_tuning.json
  load-full     run 10 RPS / 300s load test
  package       create submission zip from required final deliverables
  h100-final    run the full H100 workflow used for assignment evidence

Examples:
  ./scripts/run-full-project.sh setup
  ./scripts/run-full-project.sh stack
  ./scripts/run-full-project.sh agent
  ./scripts/run-full-project.sh health
  ./scripts/run-full-project.sh stop-all
  ./scripts/run-full-project.sh h100-final
  ./scripts/run-full-project.sh package
  CONFIG_FILE=config/profiles/h100.env ./scripts/run-full-project.sh h100-final

Notes:
  - Put HF_TOKEN and Langfuse keys in .env yourself.
  - Starting vLLM and load-full consume H100 time.
EOF
}

require_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        printf 'Install command before continuing: %s\n' "$1" >&2
        exit 1
    fi
}

compose() {
    if "$DOCKER_BIN" info >/dev/null 2>&1; then
        "$DOCKER_BIN" compose "$@"
    elif command -v sudo >/dev/null 2>&1 && sudo -n "$DOCKER_BIN" info >/dev/null 2>&1; then
        sudo "$DOCKER_BIN" compose "$@"
    else
        "$DOCKER_BIN" compose "$@"
    fi
}

read_pid_file() {
    local pid_file="$1"
    local pid
    [[ -f "$pid_file" ]] || return 1
    pid="$(cat "$pid_file")"
    [[ "$pid" =~ ^[0-9]+$ ]] || return 1
    printf '%s' "$pid"
}

stop_pid_file() {
    local name="$1"
    local pid_file="$2"
    local pattern="$3"
    local pid
    local attempt

    if pid="$(read_pid_file "$pid_file")" && kill -0 "$pid" >/dev/null 2>&1; then
        log "Stopping $name PID $pid"
        kill "$pid" >/dev/null 2>&1 || true
        for attempt in 1 2 3 4 5 6 7 8 9 10; do
            if ! kill -0 "$pid" >/dev/null 2>&1; then
                break
            fi
            sleep 1
        done
        if kill -0 "$pid" >/dev/null 2>&1; then
            log "$name PID $pid is still running after SIGTERM; check manually before forcing termination"
        fi
    else
        log "No $name PID file with a live process"
    fi

    if command -v pkill >/dev/null 2>&1; then
        pkill -f "$pattern" >/dev/null 2>&1 || true
    fi
    rm -f "$pid_file"
}

load_kv_file() {
    local env_file="$1"
    [[ -f "$env_file" ]] || return 0

    local line key value
    while IFS= read -r line || [[ -n "$line" ]]; do
        [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
        [[ "$line" == *"="* ]] || continue
        key="${line%%=*}"
        value="${line#*=}"
        key="${key#"${key%%[![:space:]]*}"}"
        key="${key%"${key##*[![:space:]]}"}"
        [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
        value="${value#"${value%%[![:space:]]*}"}"
        value="${value%"${value##*[![:space:]]}"}"
        if [[ "$value" == \"*\" && "$value" == *\" ]]; then
            value="${value:1:${#value}-2}"
        elif [[ "$value" == \'*\' && "$value" == *\' ]]; then
            value="${value:1:${#value}-2}"
        fi
        export "$key=$value"
    done < "$env_file"
}

load_env_file() {
    load_kv_file ".env"
    if [[ -n "$CONFIG_FILE" ]]; then
        load_kv_file "$CONFIG_FILE"
    fi
}

configure_backend() {
    load_env_file
    export VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:8000/v1}"
    export VLLM_MODEL="${VLLM_MODEL:-$VLLM_MODEL_FINAL}"
    export AGENT_FAST_VERIFY="${AGENT_FAST_VERIFY:-0}"
    export AGENT_MAX_ITERATIONS="${AGENT_MAX_ITERATIONS:-3}"
    export LLM_MAX_TOKENS="${LLM_MAX_TOKENS:-256}"
    log "Agent vLLM endpoint: $VLLM_BASE_URL / $VLLM_MODEL"
}

setup() {
    require_cmd "$UV_BIN"
    log "Syncing Python dependencies from uv.lock"
    "$UV_BIN" sync --frozen

    if [[ ! -f .env ]]; then
        log "Creating .env from .env.example"
        cp .env.example .env
    fi

    log "Loading BIRD data if needed"
    "$UV_BIN" run python scripts/load_data.py
}

stack() {
    require_cmd "$DOCKER_BIN"
    log "Starting Prometheus, Grafana, Langfuse, and backing services"
    compose up -d
    log "UIs: Grafana http://localhost:3000, Prometheus http://localhost:9090, Langfuse http://localhost:3001"
}

stop_stack() {
    require_cmd "$DOCKER_BIN"
    log "Stopping Prometheus, Grafana, Langfuse, and backing services"
    compose down
}

start_vllm() {
    configure_backend
    if curl -fsS http://localhost:8000/v1/models >/dev/null 2>&1; then
        log "vLLM already responds on localhost:8000"
        return
    fi
    log "Starting vLLM in background; logs/vllm.log captures output"
    nohup scripts/start_vllm.sh > "$LOG_DIR/vllm.log" 2>&1 &
    echo "$!" > "$VLLM_PID_FILE"
    log "vLLM PID $(cat "$VLLM_PID_FILE"). Wait until http://localhost:8000/v1/models responds."
}

stop_vllm() {
    stop_pid_file "vLLM" "$VLLM_PID_FILE" "vllm.entrypoints.openai.api_server"
}

stop_agent() {
    stop_pid_file "agent" "$AGENT_PID_FILE" "uvicorn agent.server:app"
}

start_agent() {
    require_cmd "$UV_BIN"
    configure_backend
    if curl -fsS "http://localhost:${AGENT_PORT}/health" >/dev/null 2>&1; then
        log "Agent already responds on localhost:${AGENT_PORT}"
        return
    fi
    log "Starting agent with ${AGENT_WORKERS} workers; structured logs go to logs/agent.log"
    nohup "$UV_BIN" run uvicorn agent.server:app \
        --host "$AGENT_HOST" \
        --port "$AGENT_PORT" \
        --workers "$AGENT_WORKERS" \
        > "$LOG_DIR/agent-uvicorn.log" 2>&1 &
    echo "$!" > "$AGENT_PID_FILE"
    sleep 3
    health
}

health() {
    log "Checking services"
    curl -fsS "http://localhost:${AGENT_PORT}/health" && printf '\n'
    curl -fsS http://localhost:8000/v1/models >/dev/null && log "vLLM OK"
    curl -fsS http://localhost:9090/-/healthy >/dev/null && log "Prometheus OK"
    curl -fsS http://localhost:3000/api/health >/dev/null && log "Grafana OK"
}

eval_baseline() {
    require_cmd "$UV_BIN"
    log "Running baseline eval"
    "$UV_BIN" run python evals/run_eval.py --out "$RESULTS_DIR/eval_baseline.json"
}

eval_after_tuning() {
    require_cmd "$UV_BIN"
    log "Running post-tuning eval"
    "$UV_BIN" run python evals/run_eval.py --out "$RESULTS_DIR/eval_after_tuning.json"
}

load_full() {
    require_cmd "$UV_BIN"
    load_env_file
    local load_out="${LOAD_OUT_FILE:-$RESULTS_DIR/load_test_${LOAD_RPS}rps_${LOAD_DURATION}s_full_agent_final.json}"
    local warmup_args=()
    if [[ "$LOAD_WARMUP_UNIQUE" == "1" || "$LOAD_WARMUP_UNIQUE" == "true" || "$LOAD_WARMUP_UNIQUE" == "yes" ]]; then
        warmup_args=(
            --warmup-unique
            --warmup-concurrency "$LOAD_WARMUP_CONCURRENCY"
            --warmup-timeout-seconds "$LOAD_WARMUP_TIMEOUT_SECONDS"
            --warmup-retries "$LOAD_WARMUP_RETRIES"
        )
        log "Warmup enabled: one request per unique scheduled DB/question before measurement"
    fi
    log "Running full load test: rps=${LOAD_RPS}, duration=${LOAD_DURATION}s"
    "$UV_BIN" run python load_test/driver.py \
        --rps "$LOAD_RPS" \
        --duration "$LOAD_DURATION" \
        --request-timeout-seconds "$LOAD_REQUEST_TIMEOUT_SECONDS" \
        "${warmup_args[@]}" \
        --out "$load_out"
}

stop_all() {
    stop_agent
    stop_vllm
    stop_stack
}

package_submission() {
    require_cmd zip
    local existing_files=()
    local absent_files=()
    local file
    local required_files=(
        REPORT.md
        infra/grafana/provisioning/dashboards/serving.json
        agent/graph.py
        agent/prompts.py
        evals/run_eval.py
        results/eval_baseline.json
        results/eval_after_tuning.json
        screenshots/vllm_manual_query.png
        screenshots/grafana_serving.png
        screenshots/langfuse_trace.png
        screenshots/langfuse_tags.png
        screenshots/grafana_eval_run.png
        screenshots/grafana_before.png
        screenshots/grafana_after.png
    )

    for file in "${required_files[@]}"; do
        if [[ -f "$file" ]]; then
            existing_files+=("$file")
        else
            absent_files+=("$file")
        fi
    done

    if (( ${#absent_files[@]} > 0 )); then
        printf 'Required final deliverables are missing:\n' >&2
        printf '  - %s\n' "${absent_files[@]}" >&2
        exit 1
    fi

    mkdir -p "$SUBMISSION_DIR"
    log "Creating submission archive: $SUBMISSION_ZIP"
    zip -q -FS "$SUBMISSION_ZIP" "${existing_files[@]}"
    log "Packaged ${#existing_files[@]} files into $SUBMISSION_ZIP"
}

h100_final() {
    CONFIG_FILE="${CONFIG_FILE:-config/profiles/h100.env}"
    setup
    stack
    start_vllm
    start_agent
    eval_baseline
    load_full
    eval_after_tuning
    log "Done. Fill Langfuse keys in .env and take required screenshots via forwarded ports."
}

default_run() {
    setup
    stack
    start_agent
    health
}

action="${1:-default}"
case "$action" in
    default) default_run ;;
    setup) setup ;;
    stack|start-observability) stack ;;
    stop-stack|stop-observability) stop_stack ;;
    vllm|start-vllm) start_vllm ;;
    stop-vllm) stop_vllm ;;
    agent|start-agent) start_agent ;;
    stop-agent) stop_agent ;;
    stop-all) stop_all ;;
    health) configure_backend; health ;;
    eval) eval_baseline ;;
    eval-after) eval_after_tuning ;;
    load-full) load_full ;;
    package) package_submission ;;
    h100-final) h100_final ;;
    help|-h|--help) usage ;;
    *)
        usage
        printf '\nUnknown action: %s\n' "$action" >&2
        exit 1
        ;;
esac
