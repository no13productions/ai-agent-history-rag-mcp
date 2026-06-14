"""Tests for installer MCP configuration generation."""

from pathlib import Path

from claude_history_rag.installer import build_mcp_server_config, project_mcp_env


def test_mcp_config_preserves_spanner_vertex_env_for_update_mode():
    """MCP registrations inherit the storage/embedding env needed for direct Spanner."""
    daemon_env = {
        "HOME": "/Users/brandon",
        "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin",
        "CLOUDSDK_CONFIG": "/Users/brandon/.config/gcloud",
        "GOOGLE_APPLICATION_CREDENTIALS": "/Users/brandon/.config/gcloud/adc.json",
        "GOOGLE_CLOUD_PROJECT": "jeeves-486102",
        "CLAUDE_HISTORY_RAG_MACHINE_ID": "mac-mini",
        "CLAUDE_HISTORY_RAG_CLIENT_NAME": "Brandon Mac Mini",
        "CLAUDE_HISTORY_RAG_STORAGE_BACKEND": "spanner",
        "CLAUDE_HISTORY_RAG_SPANNER_PROJECT": "jeeves-486102",
        "CLAUDE_HISTORY_RAG_SPANNER_INSTANCE": "jeeves-rg-spanner-prod-4d0e4c43",
        "CLAUDE_HISTORY_RAG_SPANNER_DATABASE": "ai-agent-history-rag",
        "CLAUDE_HISTORY_RAG_SPANNER_EMBEDDING_MODE": "spanner",
        "CLAUDE_HISTORY_RAG_SPANNER_EMBEDDING_MODEL_ID": "ConversationEmbeddingModel",
        "CLAUDE_HISTORY_RAG_SPANNER_DEFER_EMBEDDINGS": "true",
        "CLAUDE_HISTORY_RAG_SPANNER_BACKFILL_CONCURRENCY": "16",
        "CLAUDE_HISTORY_RAG_SPANNER_BACKFILL_BATCH_SIZE": "200",
        "CLAUDE_HISTORY_RAG_SPANNER_EMBEDDING_RPC_BATCH_SIZE": "10",
        "CLAUDE_HISTORY_RAG_SPANNER_BACKFILL_INTERVAL_SECONDS": "30",
        "CLAUDE_HISTORY_RAG_EMBEDDING_PROVIDER": "vertex",
        "CLAUDE_HISTORY_RAG_EMBEDDING_MODEL": "gemini-embedding-001",
        "CLAUDE_HISTORY_RAG_EMBEDDING_DIMENSION": "3072",
    }

    config = build_mcp_server_config(
        project_dir=Path("/repo"),
        uv_path="/usr/local/bin/uv",
        env_vars=daemon_env,
    )

    assert config["command"] == "/usr/local/bin/uv"
    assert config["args"] == ["--directory", "/repo", "run", "ai-agent-history-rag"]
    assert config["env"] == daemon_env


def test_mcp_config_filters_secret_daemon_env():
    """MCP app configs should not receive daemon secret env values."""
    config = build_mcp_server_config(
        project_dir=Path("/repo"),
        uv_path="/usr/local/bin/uv",
        env_vars={
            "CLAUDE_HISTORY_RAG_STORAGE_BACKEND": "spanner",
            "CLAUDE_HISTORY_RAG_SERVER_PSK": "server-secret",
            "CLAUDE_HISTORY_RAG_CLIENT_PSK": "client-secret",
            "CLAUDE_HISTORY_RAG_EMBEDDING_API_KEY": "embedding-secret",
            "UNRELATED_TOKEN": "token-secret",
        },
    )

    assert config["env"] == {"CLAUDE_HISTORY_RAG_STORAGE_BACKEND": "spanner"}


def test_mcp_config_overlays_explicit_client_values_on_env():
    """Explicit client arguments remain authoritative when daemon env is supplied."""
    config = build_mcp_server_config(
        project_dir=Path("/repo"),
        uv_path="/usr/local/bin/uv",
        env_vars={"CLAUDE_HISTORY_RAG_MACHINE_ID": "old"},
        server_url="http://server:4680",
        machine_id="mac-mini",
        client_name="Brandon Mac Mini",
    )

    assert config["env"]["CLAUDE_HISTORY_RAG_SERVER_URL"] == "http://server:4680"
    assert config["env"]["CLAUDE_HISTORY_RAG_MACHINE_ID"] == "mac-mini"
    assert config["env"]["CLAUDE_HISTORY_RAG_CLIENT_NAME"] == "Brandon Mac Mini"


def test_project_mcp_env_omits_empty_values():
    assert project_mcp_env(
        {
            "CLAUDE_HISTORY_RAG_STORAGE_BACKEND": "spanner",
            "CLAUDE_HISTORY_RAG_SPANNER_PROJECT": "",
        }
    ) == {"CLAUDE_HISTORY_RAG_STORAGE_BACKEND": "spanner"}
