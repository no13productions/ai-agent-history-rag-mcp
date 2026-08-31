#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DAEMON_SOURCE="$ROOT_DIR/src/claude_history_rag/daemon.py"
PASS_COUNT=0

pass() {
  PASS_COUNT=$((PASS_COUNT + 1))
  printf 'PASS %s\n' "$1"
}

fail() {
  printf 'FAIL %s\n' "$1" >&2
  exit 1
}

require_literal() {
  local file="$1"
  local literal="$2"
  local label="$3"
  grep -Fq -- "$literal" "$file" || fail "$label: missing $literal"
}

require_absent() {
  local file="$1"
  local literal="$2"
  local label="$3"
  if grep -Fq -- "$literal" "$file"; then
    fail "$label: forbidden $literal"
  fi
}

function_body() {
  local function_name="$1"
  awk -v signature="def $function_name(" '
    index($0, signature) == 1 { capture = 1 }
    capture && seen && $0 ~ /^(async )?def |^class / { exit }
    capture { print; seen = 1 }
  ' "$DAEMON_SOURCE"
}

require_body_literal() {
  local function_name="$1"
  local literal="$2"
  local body
  body="$(function_body "$function_name")"
  [[ -n "$body" ]] || fail "daemon function is missing: $function_name"
  grep -Fq -- "$literal" <<<"$body" || fail "$function_name is missing: $literal"
}

require_body_order() {
  local function_name="$1"
  shift
  local body previous=0 literal line
  body="$(function_body "$function_name")"
  [[ -n "$body" ]] || fail "daemon function is missing: $function_name"
  for literal in "$@"; do
    line="$(grep -nF -m1 -- "$literal" <<<"$body" | cut -d: -f1 || true)"
    [[ -n "$line" ]] || fail "$function_name is missing ordered step: $literal"
    ((line > previous)) || fail "$function_name reorders lifecycle step: $literal"
    previous="$line"
  done
}

[[ -r "$DAEMON_SOURCE" ]] || fail "daemon source is missing"

require_body_order is_daemon_running \
  'if not PID_FILE.exists():' \
  'os.kill(pid, 0)' \
  'if pid == os.getpid():' \
  'if not _pid_is_history_daemon(pid):' \
  'PID_FILE.unlink(missing_ok=True)' \
  'return True, pid'
require_body_literal is_daemon_running 'except (ValueError, ProcessLookupError, PermissionError):'
pass "PID ownership rejects stale or reused processes"

require_body_order _find_history_mcp_worker_pids \
  'current_pid = os.getpid()' \
  'if pid in {None, current_pid}:' \
  'if not _command_belongs_to_project(command, root):' \
  'if "ai-agent-history-rag-daemon" in command or "claude_history_rag.daemon" in command:' \
  'if "ai-agent-history-rag" in command or "claude_history_rag.__main__" in command:' \
  'worker_pids.append(int(pid))' \
  'return sorted(set(worker_pids))'
pass "worker cleanup remains same-checkout and daemon-excluding"

require_body_order terminate_daemon_process \
  'os.kill(pid, signal.SIGTERM)' \
  'if _wait_for_pid_exit(pid, timeout_seconds):' \
  'os.kill(pid, signal.SIGKILL)' \
  'if _wait_for_pid_exit(pid, kill_timeout_seconds):'
require_body_literal terminate_daemon_process 'except PermissionError:'
require_body_literal terminate_daemon_process 'return False'
pass "daemon termination remains bounded and fail closed"

require_body_order terminate_worker_processes \
  'targets = sorted({pid for pid in pids if pid != os.getpid()})' \
  'os.kill(pid, signal.SIGTERM)' \
  'survivors = [pid for pid in targets if not _wait_for_pid_exit(pid, timeout_seconds)]' \
  'os.kill(pid, signal.SIGKILL)' \
  'remaining = [pid for pid in survivors if not _wait_for_pid_exit(pid, kill_timeout_seconds)]' \
  'return not remaining'
require_body_literal terminate_worker_processes 'except PermissionError:'
pass "worker termination remains bounded and fail closed"

require_body_order cmd_start \
  'is_running, pid = is_daemon_running()' \
  'if is_running:' \
  'return 0' \
  'return _run_foreground_daemon()'
pass "human start remains idempotent"

require_body_order cmd_supervise \
  'is_running, pid = is_daemon_running()' \
  'if not terminate_daemon_process(pid):' \
  'worker_pids = _find_history_mcp_worker_pids()' \
  'if not terminate_worker_processes(worker_pids):' \
  'return _run_foreground_daemon()'
require_body_literal cmd_supervise 'return 1'
pass "supervisor replaces old owners before foreground run"

require_body_order cmd_stop \
  'is_running, pid = is_daemon_running()' \
  'if not is_running:' \
  'terminate_daemon_process(pid, timeout_seconds=15.0, kill_timeout_seconds=5.0)' \
  'print("Daemon stopped")' \
  'return 0'
pass "manual stop uses the shared bounded termination path"

service_files=(
  "$ROOT_DIR/scripts/com.ai-agent-history-rag.daemon.plist.template"
  "$ROOT_DIR/scripts/install-launchd.sh"
  "$ROOT_DIR/scripts/install-systemd.sh"
  "$ROOT_DIR/scripts/install-windows.ps1"
  "$ROOT_DIR/scripts/ai-agent-history-rag.service"
)
for service_file in "${service_files[@]}"; do
  [[ -r "$service_file" ]] || fail "service-manager source is missing: $service_file"
  require_literal "$service_file" 'ai-agent-history-rag-daemon' "service manager must select daemon"
  require_literal "$service_file" 'supervise' "service manager must select supervised lifecycle"
  require_absent "$service_file" 'ai-agent-history-rag-daemon start' "service manager must not select human start"
done
pass "all service-manager sources select supervise"

plist="$ROOT_DIR/scripts/com.ai-agent-history-rag.daemon.plist.template"
required_plist_literals=(
  'CLAUDE_HISTORY_RAG_RUNTIME_CONTRACT'
  'production'
  'CLAUDE_HISTORY_RAG_STORAGE_BACKEND'
  'spanner'
  'gemini-embedding-001'
  '3072'
  'CLAUDE_HISTORY_RAG_CREDENTIALS_SOURCE'
  'application_default'
  'CLAUDE_HISTORY_RAG_CREDENTIALS_PROFILE'
  'impersonated_service_account'
  'CLAUDE_HISTORY_RAG_CREDENTIALS_IDENTITY'
  'GOOGLE_APPLICATION_CREDENTIALS'
  '__GCP_PROJECT__'
  '__SPANNER_INSTANCE__'
)
for literal in "${required_plist_literals[@]}"; do
  require_literal "$plist" "$literal" "launchd template production contract"
done
require_absent "$plist" 'CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE' "launchd template must reject credential override"
require_absent "$plist" 'CLAUDE_RAG_' "launchd template must reject legacy variables"
pass "launchd template pins production Spanner and keyless ADC shape"

identity_files=(
  "$ROOT_DIR/scripts/com.ai-agent-history-rag.daemon.plist.template"
  "$ROOT_DIR/scripts/install-launchd.sh"
)
for identity_file in "${identity_files[@]}"; do
  require_absent "$identity_file" 'jeeves-' "public service source must not publish deployment identity"
  require_absent "$identity_file" 'alfred-sa-key' "public service source must not publish key coordinates"
  require_absent "$identity_file" '/Users/brandon' "public service source must not publish host paths"
done
pass "public service sources contain no live deployment coordinates"

printf 'PASS total=%d\n' "$PASS_COUNT"
