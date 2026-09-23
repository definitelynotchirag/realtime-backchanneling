#!/usr/bin/env bash
# Start or stop the local LiveKit worker, FastAPI service, and dashboard together.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="$ROOT_DIR/.runtime"
LOG_DIR="$RUNTIME_DIR/logs"
PID_DIR="$RUNTIME_DIR/pids"

WORKER_PID="$PID_DIR/worker.pid"
API_PID="$PID_DIR/api.pid"
DASHBOARD_PID="$PID_DIR/dashboard.pid"

API_PORT="${API_PORT:-8000}"
DASHBOARD_PORT="${DASHBOARD_PORT:-3001}"

log() {
  printf '%s\n' "$*"
}

is_running() {
  local pid_file="$1"
  [[ -f "$pid_file" ]] && kill -0 "$(<"$pid_file")" 2>/dev/null
}

require_available_port() {
  local port="$1"
  if command -v lsof >/dev/null 2>&1 && lsof -tiTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
    log "Port $port is already in use. Stop that service first, or choose another port."
    exit 1
  fi
}

require_no_unmanaged_worker() {
  local managed_pid=""
  if is_running "$WORKER_PID"; then
    managed_pid="$(<"$WORKER_PID")"
  fi

  local pid
  while IFS= read -r pid; do
    [[ -z "$pid" || "$pid" == "$managed_pid" ]] && continue
    log "A different Blue Machines worker is already running (PID $pid)."
    log "Stop it from the terminal where it was started before using this script."
    exit 1
  done < <(pgrep -f "[b]lue-machines-agent dev" || true)
}

start_process() {
  local name="$1"
  local pid_file="$2"
  local log_file="$3"
  shift 3

  if is_running "$pid_file"; then
    log "$name is already running (PID $(<"$pid_file"))."
    return
  fi

  rm -f "$pid_file"
  (
    cd "$ROOT_DIR"
    exec "$@"
  ) >"$log_file" 2>&1 &
  echo "$!" >"$pid_file"
  log "Started $name (PID $(<"$pid_file"))."
}

stop_process() {
  local name="$1"
  local pid_file="$2"
  if ! [[ -f "$pid_file" ]]; then
    return
  fi

  local pid
  pid="$(<"$pid_file")"
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid"
    log "Stopped $name (PID $pid)."
  fi
  rm -f "$pid_file"
}

wait_for_api() {
  local attempt
  for attempt in {1..20}; do
    if curl --silent --fail "http://127.0.0.1:$API_PORT/health" >/dev/null; then
      return
    fi
    sleep 0.25
  done
  log "The API did not become healthy. Check $LOG_DIR/api.log"
  exit 1
}

start() {
  mkdir -p "$LOG_DIR" "$PID_DIR"
  require_no_unmanaged_worker
  if ! is_running "$API_PID"; then
    require_available_port "$API_PORT"
  fi
  if ! is_running "$DASHBOARD_PID"; then
    require_available_port "$DASHBOARD_PORT"
  fi

  start_process "LiveKit worker" "$WORKER_PID" "$LOG_DIR/worker.log" \
    "$ROOT_DIR/.venv/bin/blue-machines-agent" dev
  start_process "FastAPI service" "$API_PID" "$LOG_DIR/api.log" \
    "$ROOT_DIR/.venv/bin/uvicorn" blue_machines_baseline.api:app --host 127.0.0.1 --port "$API_PORT"
  start_process "Dashboard" "$DASHBOARD_PID" "$LOG_DIR/dashboard.log" \
    bash -c 'cd "$1" && shift && exec "$@"' -- "$ROOT_DIR/dashboard" \
    "$ROOT_DIR/dashboard/node_modules/.bin/next" dev --hostname 127.0.0.1 --port "$DASHBOARD_PORT"

  wait_for_api
  log ""
  log "Stack is ready:"
  log "  Dashboard: http://127.0.0.1:$DASHBOARD_PORT"
  log "  API:       http://127.0.0.1:$API_PORT/health"
  log "  Logs:      $LOG_DIR"
  log ""
  log "Stop only these managed processes with: ./scripts/dev-stack.sh stop"
}

status() {
  local name pid_file
  for name in "LiveKit worker:$WORKER_PID" "FastAPI service:$API_PID" "Dashboard:$DASHBOARD_PID"; do
    local display_name="${name%%:*}"
    local display_pid_file="${name#*:}"
    if is_running "$display_pid_file"; then
      log "$display_name: running (PID $(<"$display_pid_file"))"
    else
      log "$display_name: stopped"
    fi
  done
}

case "${1:-start}" in
  start) start ;;
  stop)
    stop_process "Dashboard" "$DASHBOARD_PID"
    stop_process "FastAPI service" "$API_PID"
    stop_process "LiveKit worker" "$WORKER_PID"
    ;;
  restart)
    "$0" stop
    "$0" start
    ;;
  status) status ;;
  *)
    log "Usage: $0 {start|stop|restart|status}"
    exit 2
    ;;
esac
