#!/usr/bin/env bash
set -euo pipefail

PROTOCOL_VERSION="2025-06-18"
DAEMON_BASE_URL="http://127.0.0.1:4680"
DAEMON_PSK=""
CURL_BIN=""

contract_fail() {
  printf 'production runtime contract invalid: %s\n' "$1" >&2
  exit 64
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || contract_fail "required native command is unavailable: $1"
}

required_value() {
  local name="$1"
  [[ -n "${!name-}" ]] || contract_fail "$name must be set"
}

exact_value() {
  local name="$1"
  local expected="$2"
  [[ "${!name-}" == "$expected" ]] || contract_fail "$name does not match the production contract"
}

file_mode() {
  local path="$1"
  stat -c '%a' -- "$path" 2>/dev/null || stat -f '%Lp' -- "$path" 2>/dev/null
}

file_owner() {
  local path="$1"
  stat -c '%u' -- "$path" 2>/dev/null || stat -f '%u' -- "$path" 2>/dev/null
}

secure_regular_file() {
  local path="$1"
  local label="$2"
  [[ ! -L "$path" ]] || contract_fail "$label must not be a symlink"
  [[ -f "$path" ]] || contract_fail "$label must be a regular file"
  local mode owner
  mode="$(file_mode "$path")" || contract_fail "$label permissions cannot be inspected"
  owner="$(file_owner "$path")" || contract_fail "$label owner cannot be inspected"
  [[ "$mode" =~ ^[0-7]{3,4}$ ]] || contract_fail "$label permissions are invalid"
  local mode_value=$((8#$mode))
  (( (mode_value & 077) == 0 )) || contract_fail "$label must be owner-readable only"
  [[ "$owner" == "$(id -u)" ]] || contract_fail "$label must be owned by the current user"
}

contains_private_key_fields() {
  local adc_path="$1"
  jq -e '
    [.. | objects | select(has("private_key") or has("private_key_id"))]
    | length == 0
  ' "$adc_path" >/dev/null 2>&1
}

validate_impersonated_adc() {
  local adc_path="$1"
  local expected_identity="$2"
  [[ -n "$adc_path" ]] || contract_fail "GOOGLE_APPLICATION_CREDENTIALS must select the impersonated ADC profile"
  secure_regular_file "$adc_path" "GOOGLE_APPLICATION_CREDENTIALS"
  local bytes
  bytes="$(wc -c <"$adc_path" | tr -d ' ')"
  if [[ ! "$bytes" =~ ^[0-9]+$ ]] || (( bytes > 1048576 )); then
    contract_fail "GOOGLE_APPLICATION_CREDENTIALS exceeds the bounded JSON size"
  fi
  jq -e 'type == "object"' "$adc_path" >/dev/null 2>&1 || contract_fail "GOOGLE_APPLICATION_CREDENTIALS must contain a JSON object"
  [[ "$(jq -r '.type // empty' "$adc_path")" == "impersonated_service_account" ]] || contract_fail "GOOGLE_APPLICATION_CREDENTIALS must be an impersonated_service_account ADC profile"
  contains_private_key_fields "$adc_path" || contract_fail "GOOGLE_APPLICATION_CREDENTIALS must not contain private key material"

  local source_type
  source_type="$(jq -r '.source_credentials.type // empty' "$adc_path")"
  [[ "$source_type" == "authorized_user" ]] || contract_fail "impersonated ADC source_credentials must be authorized_user"
  jq -e '
    (.source_credentials | type == "object") and
    (.source_credentials.client_id | type == "string" and length > 0) and
    (.source_credentials.client_secret | type == "string" and length > 0) and
    (.source_credentials.refresh_token | type == "string" and length > 0)
  ' "$adc_path" >/dev/null 2>&1 || contract_fail "impersonated ADC authorized_user source is incomplete"
  jq -e '(.delegates == null) or (.delegates == [])' "$adc_path" >/dev/null 2>&1 || contract_fail "impersonated ADC delegates must be absent"

  local expected_url actual_url
  expected_url="https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/${expected_identity}:generateAccessToken"
  actual_url="$(jq -r '.service_account_impersonation_url // empty' "$adc_path")"
  [[ "$actual_url" == "$expected_url" ]] || contract_fail "impersonated ADC target identity does not match credentials_identity"
}

validate_source_paths() {
  local home="$HOME"
  [[ "$home" == /* ]] || contract_fail "HOME must be absolute"
  local checks=(
    "CLAUDE_HISTORY_RAG_PROJECTS_PATH|$home/.claude/projects"
    "CLAUDE_HISTORY_RAG_CODEX_SESSIONS_PATH|$home/.codex/sessions"
    "CLAUDE_HISTORY_RAG_GEMINI_SESSIONS_PATH|$home/.gemini/tmp"
    "CLAUDE_HISTORY_RAG_ANTIGRAVITY_SESSIONS_PATH|$home/.gemini/antigravity"
    "CLAUDE_HISTORY_RAG_CHATGPT_EXPORTS_PATH|$home/.claude-history-rag/imports/chatgpt"
    "CLAUDE_HISTORY_RAG_CLAUDE_APP_EXPORTS_PATH|$home/.claude-history-rag/imports/claude-app"
  )
  local check name expected actual
  for check in "${checks[@]}"; do
    name="${check%%|*}"
    expected="${check#*|}"
    actual="${!name-$expected}"
    [[ "$actual" == "$expected" ]] || contract_fail "$name does not match the production source-path contract"
  done
}

validate_production_contract() {
  require_command jq
  required_value HOME
  exact_value CLAUDE_HISTORY_RAG_RUNTIME_CONTRACT production
  exact_value CLAUDE_HISTORY_RAG_STORAGE_BACKEND spanner
  required_value CLAUDE_HISTORY_RAG_SPANNER_PROJECT
  required_value CLAUDE_HISTORY_RAG_SPANNER_INSTANCE
  required_value CLAUDE_HISTORY_RAG_SPANNER_DATABASE
  exact_value CLAUDE_HISTORY_RAG_SPANNER_EMBEDDING_MODE spanner
  exact_value CLAUDE_HISTORY_RAG_SPANNER_EMBEDDING_MODEL_ID ConversationEmbeddingModel
  exact_value CLAUDE_HISTORY_RAG_EMBEDDING_PROVIDER vertex
  exact_value CLAUDE_HISTORY_RAG_EMBEDDING_MODEL gemini-embedding-001
  exact_value CLAUDE_HISTORY_RAG_EMBEDDING_DIMENSION 3072
  exact_value CLAUDE_HISTORY_RAG_STATUS_SERVER_HOST 127.0.0.1
  exact_value CLAUDE_HISTORY_RAG_STATUS_SERVER_PORT 4680
  exact_value CLAUDE_HISTORY_RAG_CREDENTIALS_SOURCE application_default
  required_value CLAUDE_HISTORY_RAG_CREDENTIALS_IDENTITY

  [[ "$CLAUDE_HISTORY_RAG_CREDENTIALS_IDENTITY" =~ ^[a-z][a-z0-9-]{4,28}[a-z0-9]@[a-z][a-z0-9-]{4,28}[a-z0-9]\.iam\.gserviceaccount\.com$ ]] || contract_fail "CLAUDE_HISTORY_RAG_CREDENTIALS_IDENTITY must be a service-account email"
  [[ -z "${CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE-}" ]] || contract_fail "CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE must be unset"
  [[ -z "${CLAUDE_HISTORY_RAG_DB_PATH-}" ]] || contract_fail "CLAUDE_HISTORY_RAG_DB_PATH must be unset"
  validate_source_paths

  case "${CLAUDE_HISTORY_RAG_CREDENTIALS_PROFILE-}" in
    impersonated_service_account)
      validate_impersonated_adc "${GOOGLE_APPLICATION_CREDENTIALS-}" "$CLAUDE_HISTORY_RAG_CREDENTIALS_IDENTITY"
      ;;
    attached_service_account)
      [[ -z "${GOOGLE_APPLICATION_CREDENTIALS-}" ]] || contract_fail "attached_service_account must not select a credential file"
      ;;
    *)
      contract_fail "CLAUDE_HISTORY_RAG_CREDENTIALS_PROFILE must select an admitted production profile"
      ;;
  esac

  case "${CLAUDE_HISTORY_RAG_AUTH_ENABLED-true}" in
    true|false) ;;
    *) contract_fail "CLAUDE_HISTORY_RAG_AUTH_ENABLED must be true or false" ;;
  esac
}

load_daemon_psk() {
  DAEMON_PSK=""
  [[ "${CLAUDE_HISTORY_RAG_AUTH_ENABLED-true}" == "true" ]] || return 0
  if [[ -n "${CLAUDE_HISTORY_RAG_SERVER_PSK-}" ]]; then
    DAEMON_PSK="$CLAUDE_HISTORY_RAG_SERVER_PSK"
  else
    local auth_path="${CLAUDE_HISTORY_RAG_AUTH_STATE_PATH:-$HOME/.claude-history-rag/auth.json}"
    secure_regular_file "$auth_path" "CLAUDE_HISTORY_RAG_AUTH_STATE_PATH"
    local bytes
    bytes="$(wc -c <"$auth_path" | tr -d ' ')"
    if [[ ! "$bytes" =~ ^[0-9]+$ ]] || (( bytes > 1048576 )); then
      contract_fail "auth state exceeds the bounded JSON size"
    fi
    jq -e 'type == "object"' "$auth_path" >/dev/null 2>&1 || contract_fail "auth state must contain a JSON object"
    DAEMON_PSK="$(jq -r '.active.key_plain // empty' "$auth_path")"
  fi
  [[ -n "$DAEMON_PSK" ]] || contract_fail "daemon authentication is enabled but no active PSK is available"
  [[ "$DAEMON_PSK" != *$'\n'* && "$DAEMON_PSK" != *$'\r'* ]] || contract_fail "daemon PSK contains an invalid control character"
}

daemon_request() {
  local method="$1"
  local path="$2"
  local request_body="${3-}"
  local combined
  local common=(
    --silent
    --show-error
    --noproxy '*'
    --proto '=http'
    --max-redirs 0
    --connect-timeout 1
    --max-time 120
    --write-out $'\n%{http_code}'
  )
  local auth_args=()
  if [[ -n "$DAEMON_PSK" ]]; then
    auth_args=(--header @/dev/fd/3)
  fi

  if [[ "$method" == "POST" ]]; then
    if [[ -n "$DAEMON_PSK" ]]; then
      if ! combined="$(printf '%s' "$request_body" | "$CURL_BIN" "${common[@]}" "${auth_args[@]}" --header 'Content-Type: application/json' --request POST --data-binary @- "$DAEMON_BASE_URL$path" 3<<<"Authorization: Bearer $DAEMON_PSK")"; then
        return 1
      fi
    elif ! combined="$(printf '%s' "$request_body" | "$CURL_BIN" "${common[@]}" --header 'Content-Type: application/json' --request POST --data-binary @- "$DAEMON_BASE_URL$path")"; then
      return 1
    fi
  elif [[ -n "$DAEMON_PSK" ]]; then
    if ! combined="$("$CURL_BIN" "${common[@]}" "${auth_args[@]}" --request GET "$DAEMON_BASE_URL$path" 3<<<"Authorization: Bearer $DAEMON_PSK")"; then
      return 1
    fi
  elif ! combined="$("$CURL_BIN" "${common[@]}" --request GET "$DAEMON_BASE_URL$path")"; then
    return 1
  fi

  local status="${combined##*$'\n'}"
  DAEMON_RESPONSE="${combined%$'\n'*}"
  [[ "$status" =~ ^2[0-9][0-9]$ ]] || return 1
  jq -e 'type == "object"' <<<"$DAEMON_RESPONSE" >/dev/null 2>&1 || return 1
}

tool_catalog() {
  jq -cn '
    {
      tools: [
        {
          name: "search_conversations",
          description: "Search indexed conversation history through the gated local daemon.",
          inputSchema: {
            type: "object",
            properties: {
              query: {type: "string", minLength: 1, maxLength: 10000},
              project_filter: {type: ["string", "null"]},
              date_from: {type: ["string", "null"]},
              date_to: {type: ["string", "null"]},
              limit: {type: "integer", minimum: 1, maximum: 50, default: 5},
              use_hybrid: {type: "boolean", default: true},
              enable_analysis: {type: "boolean", default: true},
              enable_synthesis: {type: "boolean", default: false},
              include_debug: {type: "boolean", default: false}
            },
            required: ["query"],
            additionalProperties: false
          }
        },
        {
          name: "search_file_changes",
          description: "Search indexed file changes through the gated local daemon.",
          inputSchema: {
            type: "object",
            properties: {
              file_path: {type: ["string", "null"]},
              query: {type: ["string", "null"]},
              project_filter: {type: ["string", "null"]},
              operation_filter: {type: ["string", "null"], enum: ["edit", "write", null]},
              date_from: {type: ["string", "null"]},
              date_to: {type: ["string", "null"]},
              limit: {type: "integer", minimum: 1, maximum: 50, default: 10}
            },
            additionalProperties: false
          }
        },
        {
          name: "get_session_summary",
          description: "Retrieve recent indexed session summaries through the gated local daemon.",
          inputSchema: {
            type: "object",
            properties: {
              session_id: {type: ["string", "null"]},
              project_filter: {type: ["string", "null"]},
              count: {type: "integer", minimum: 1, maximum: 20, default: 1}
            },
            additionalProperties: false
          }
        },
        {
          name: "get_index_status",
          description: "Get index health from the gated local daemon.",
          inputSchema: {type: "object", properties: {}, additionalProperties: false}
        },
        {
          name: "get_server_status",
          description: "Get daemon health and runtime status.",
          inputSchema: {
            type: "object",
            properties: {detail_level: {type: "string", enum: ["basic", "full"], default: "basic"}},
            additionalProperties: false
          }
        }
      ]
    }
  '
}

emit_result() {
  local id="$1"
  local result="$2"
  jq -cn --argjson id "$id" --argjson result "$result" '{jsonrpc:"2.0", id:$id, result:$result}'
}

emit_error() {
  local id="$1"
  local code="$2"
  local message="$3"
  jq -cn --argjson id "$id" --argjson code "$code" --arg message "$message" '{jsonrpc:"2.0", id:$id, error:{code:$code,message:$message}}'
}

emit_tool_error() {
  local id="$1"
  local message="$2"
  local result
  result="$(jq -cn --arg message "$message" '{content:[{type:"text",text:$message}],isError:true}')"
  emit_result "$id" "$result"
}

normalize_search_arguments() {
  jq -ce '
    if type != "object" or (.query | type) != "string" or (.query | length) < 1 or (.query | length) > 10000 then error("invalid query")
    elif (if has("limit") then (.limit | type != "number" or . < 1 or . > 50 or floor != .) else false end) then error("invalid limit")
    elif (if has("use_hybrid") then (.use_hybrid | type != "boolean") else false end) then error("invalid use_hybrid")
    elif (if has("enable_analysis") then (.enable_analysis | type != "boolean") else false end) then error("invalid enable_analysis")
    elif (if has("enable_synthesis") then (.enable_synthesis | type != "boolean") else false end) then error("invalid enable_synthesis")
    elif (if has("include_debug") then (.include_debug | type != "boolean") else false end) then error("invalid include_debug")
    else {
      query: .query,
      limit: (if has("limit") then .limit else 5 end),
      project_filter: (.project_filter // null),
      date_from: (.date_from // null),
      date_to: (.date_to // null),
      use_hybrid: (if has("use_hybrid") then .use_hybrid else true end),
      enable_analysis: (if has("enable_analysis") then .enable_analysis else true end),
      enable_synthesis: (if has("enable_synthesis") then .enable_synthesis else false end),
      include_debug: (if has("include_debug") then .include_debug else false end)
    } end
  '
}

normalize_file_arguments() {
  jq -ce '
    if type != "object" then error("invalid arguments")
    elif (if has("limit") then (.limit | type != "number" or . < 1 or . > 50 or floor != .) else false end) then error("invalid limit")
    elif ((.operation_filter // null) as $operation | ($operation != null and $operation != "edit" and $operation != "write")) then error("invalid operation_filter")
    else {
      file_path: (.file_path // null),
      query: (if (.query // null) == null and (.file_path // null) == null then "file changes modifications edits" else (.query // null) end),
      project_filter: (.project_filter // null),
      operation_filter: (.operation_filter // null),
      date_from: (.date_from // null),
      date_to: (.date_to // null),
      limit: (if has("limit") then .limit else 10 end)
    } end
  '
}

normalize_session_arguments() {
  jq -ce '
    if type != "object" then error("invalid arguments")
    elif (if has("count") then (.count | type != "number" or . < 1 or . > 20 or floor != .) else false end) then error("invalid count")
    else {
      session_id: (.session_id // null),
      project_filter: (.project_filter // null),
      count: (if has("count") then .count else 1 end)
    } end
  '
}

handle_tool_call() {
  local id="$1"
  local request="$2"
  local name arguments body path method detail result
  name="$(jq -r '.params.name // empty' <<<"$request")"
  arguments="$(jq -c '.params.arguments // {}' <<<"$request")"
  case "$name" in
    search_conversations)
      if ! body="$(normalize_search_arguments <<<"$arguments" 2>/dev/null)"; then
        emit_tool_error "$id" "Invalid search_conversations arguments"
        return
      fi
      method=POST
      path=/api/search
      ;;
    search_file_changes)
      if ! body="$(normalize_file_arguments <<<"$arguments" 2>/dev/null)"; then
        emit_tool_error "$id" "Invalid search_file_changes arguments"
        return
      fi
      method=POST
      path=/api/search/files
      ;;
    get_session_summary)
      if ! body="$(normalize_session_arguments <<<"$arguments" 2>/dev/null)"; then
        emit_tool_error "$id" "Invalid get_session_summary arguments"
        return
      fi
      method=POST
      path=/api/sessions
      ;;
    get_index_status)
      [[ "$arguments" == "{}" ]] || { emit_tool_error "$id" "Invalid get_index_status arguments"; return; }
      method=GET
      path='/status?detail=full'
      body=""
      ;;
    get_server_status)
      detail="$(jq -r '.detail_level // "basic"' <<<"$arguments")"
      [[ "$detail" == "basic" || "$detail" == "full" ]] || { emit_tool_error "$id" "Invalid get_server_status arguments"; return; }
      method=GET
      path="/status?detail=$detail"
      body=""
      ;;
    *)
      emit_tool_error "$id" "Unknown tool"
      return
      ;;
  esac

  if ! daemon_request "$method" "$path" "$body"; then
    emit_tool_error "$id" "Local daemon request failed"
    return
  fi
  result="$(jq -cn --argjson data "$DAEMON_RESPONSE" '{content:[{type:"text",text:($data|tojson)}],structuredContent:$data,isError:false}')"
  emit_result "$id" "$result"
}

serve_stdio() {
  CURL_BIN="$(command -v curl)" || contract_fail "required native command is unavailable: curl"
  load_daemon_psk
  local line request method id has_id result
  local initialize_seen=false
  local client_ready=false
  while IFS= read -r line; do
    [[ -n "$line" ]] || continue
    if (( ${#line} > 1048576 )); then
      emit_error null -32600 "Request exceeds the bounded message size"
      continue
    fi
    if ! request="$(jq -c . <<<"$line" 2>/dev/null)"; then
      emit_error null -32700 "Parse error"
      continue
    fi
    if [[ "$(jq -r '.jsonrpc // empty' <<<"$request")" != "2.0" ]]; then
      emit_error null -32600 "Invalid Request"
      continue
    fi
    method="$(jq -r '.method // empty' <<<"$request")"
    has_id=false
    if jq -e 'has("id")' <<<"$request" >/dev/null; then
      has_id=true
      id="$(jq -c '.id' <<<"$request")"
    else
      id=null
    fi
    case "$method" in
      initialize)
        if [[ "$has_id" == true ]]; then
          if [[ "$initialize_seen" == true ]]; then
            emit_error "$id" -32600 "Initialize may be sent only once"
          else
            initialize_seen=true
            result="$(jq -cn --arg protocol "$PROTOCOL_VERSION" '{protocolVersion:$protocol,capabilities:{tools:{listChanged:false}},serverInfo:{name:"ai-agent-history-rag",version:"native-gate-1"}}')"
            emit_result "$id" "$result"
          fi
        fi
        ;;
      notifications/initialized)
        [[ "$initialize_seen" == true ]] && client_ready=true
        ;;
      notifications/cancelled)
        ;;
      ping)
        [[ "$has_id" == true ]] && emit_result "$id" '{}'
        ;;
      tools/list)
        if [[ "$has_id" == true ]]; then
          if [[ "$client_ready" == true ]]; then emit_result "$id" "$(tool_catalog)"; else emit_error "$id" -32002 "Server is not initialized"; fi
        fi
        ;;
      tools/call)
        if [[ "$has_id" == true ]]; then
          if [[ "$client_ready" == true ]]; then handle_tool_call "$id" "$request"; else emit_error "$id" -32002 "Server is not initialized"; fi
        fi
        ;;
      *)
        [[ "$has_id" == true ]] && emit_error "$id" -32601 "Method not found"
        ;;
    esac
  done
}

projected_environment() {
  local keys=(
    HOME PATH CLOUDSDK_CONFIG GOOGLE_CLOUD_PROJECT GOOGLE_APPLICATION_CREDENTIALS
    CLAUDE_HISTORY_RAG_RUNTIME_CONTRACT CLAUDE_HISTORY_RAG_CREDENTIALS_SOURCE
    CLAUDE_HISTORY_RAG_CREDENTIALS_PROFILE CLAUDE_HISTORY_RAG_CREDENTIALS_IDENTITY
    CLAUDE_HISTORY_RAG_STORAGE_BACKEND CLAUDE_HISTORY_RAG_SPANNER_PROJECT
    CLAUDE_HISTORY_RAG_SPANNER_INSTANCE CLAUDE_HISTORY_RAG_SPANNER_DATABASE
    CLAUDE_HISTORY_RAG_SPANNER_EMBEDDING_MODE CLAUDE_HISTORY_RAG_SPANNER_EMBEDDING_MODEL_ID
    CLAUDE_HISTORY_RAG_EMBEDDING_PROVIDER CLAUDE_HISTORY_RAG_EMBEDDING_MODEL
    CLAUDE_HISTORY_RAG_EMBEDDING_DIMENSION CLAUDE_HISTORY_RAG_STATUS_SERVER_HOST
    CLAUDE_HISTORY_RAG_STATUS_SERVER_PORT CLAUDE_HISTORY_RAG_AUTH_ENABLED
    CLAUDE_HISTORY_RAG_AUTH_STATE_PATH CLAUDE_HISTORY_RAG_PROJECTS_PATH
    CLAUDE_HISTORY_RAG_CODEX_SESSIONS_PATH CLAUDE_HISTORY_RAG_GEMINI_SESSIONS_PATH
    CLAUDE_HISTORY_RAG_ANTIGRAVITY_SESSIONS_PATH CLAUDE_HISTORY_RAG_CHATGPT_EXPORTS_PATH
    CLAUDE_HISTORY_RAG_CLAUDE_APP_EXPORTS_PATH
  )
  local output='{}' key value
  for key in "${keys[@]}"; do
    value="${!key-}"
    [[ -n "$value" ]] || continue
    if [[ "$key" == "GOOGLE_APPLICATION_CREDENTIALS" && "$CLAUDE_HISTORY_RAG_CREDENTIALS_PROFILE" != "impersonated_service_account" ]]; then
      continue
    fi
    output="$(jq -cn --argjson current "$output" --arg key "$key" --arg value "$value" '$current + {($key):$value}')"
  done
  printf '%s\n' "$output"
}

install_json_config() {
  local config_path="$1"
  local binary_path="$2"
  [[ "$config_path" == /* ]] || contract_fail "client config path must be absolute"
  [[ "$binary_path" == /* ]] || contract_fail "native MCP command path must be absolute"
  [[ ! -L "$binary_path" && -f "$binary_path" && -x "$binary_path" ]] || contract_fail "native MCP command must be a regular executable"

  local config_dir
  config_dir="$(dirname "$config_path")"
  mkdir -p "$config_dir"
  if [[ -e "$config_path" ]]; then
    secure_regular_file "$config_path" "client MCP config"
    jq -e 'type == "object"' "$config_path" >/dev/null 2>&1 || contract_fail "client MCP config must contain a JSON object"
  else
    printf '{}\n' >"$config_path"
    chmod 600 "$config_path"
  fi
  local env_json temp_path
  env_json="$(projected_environment)"
  temp_path="$(mktemp "$config_dir/.history-rag-mcp.XXXXXX")"
  chmod 600 "$temp_path"
  if ! jq --arg command "$binary_path" --argjson projected "$env_json" '
    .mcpServers = (.mcpServers // {}) |
    .mcpServers["ai-agent-history-rag"] = {command:$command,args:[],env:$projected}
  ' "$config_path" >"$temp_path"; then
    rm -f "$temp_path"
    contract_fail "client MCP config update failed"
  fi
  mv "$temp_path" "$config_path"
  chmod 600 "$config_path"
}

usage() {
  printf '%s\n' \
    'usage: history-rag-mcp-native.sh [--validate-only]' \
    '       history-rag-mcp-native.sh --install-json ABSOLUTE_CONFIG ABSOLUTE_COMMAND'
}

main() {
  validate_production_contract # GATE_BEFORE_NETWORK_OR_HANDLER
  case "${1-}" in
    --validate-only)
      [[ $# -eq 1 ]] || { usage >&2; exit 64; }
      load_daemon_psk
      ;;
    --install-json)
      [[ $# -eq 3 ]] || { usage >&2; exit 64; }
      install_json_config "$2" "$3"
      ;;
    "")
      serve_stdio
      ;;
    *)
      usage >&2
      exit 64
      ;;
  esac
}

main "$@"
