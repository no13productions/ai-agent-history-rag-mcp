// Package runtimeconfig owns the native History-RAG production process shape.
// The contract is deliberately closed: production is Spanner + Vertex on the
// established loopback status endpoint and one explicit keyless credential
// selector. Deployment identities remain required inputs, never source defaults.
package runtimeconfig

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/no13productions/ai-agent-history-rag-mcp/internal/gcpauth"
)

const (
	ProductionContract                = "production"
	ProductionStorageBackend          = "spanner"
	ProductionSpannerEmbeddingMode    = "spanner"
	ProductionSpannerEmbeddingModelID = "ConversationEmbeddingModel"
	ProductionEmbeddingProvider       = "vertex"
	ProductionEmbeddingModel          = "gemini-embedding-001"
	ProductionEmbeddingDimension      = 3072
	ProductionStatusServerHost        = "127.0.0.1"
	ProductionStatusServerPort        = 4680
)

type SpannerTarget struct {
	Project  string
	Instance string
	Database string
}

func (s SpannerTarget) ResourceName() string {
	return fmt.Sprintf("projects/%s/instances/%s/databases/%s", s.Project, s.Instance, s.Database)
}

type Config struct {
	RuntimeContract         string
	Spanner                 SpannerTarget
	StorageBackend          string
	SpannerEmbeddingMode    string
	SpannerEmbeddingModelID string
	EmbeddingProvider       string
	EmbeddingModel          string
	EmbeddingDimension      int
	StatusServerHost        string
	StatusServerPort        int
	GoogleCredentials       gcpauth.Selector
}

// LoadProduction reads and validates the exact production contract. getenv is
// injected so tests and launch adapters can prove the complete input surface.
func LoadProduction(getenv func(string) string) (Config, error) {
	if getenv == nil {
		return Config{}, fmt.Errorf("history_rag_runtime: environment reader is required")
	}
	embeddingDimension, err := parseRequiredInt(getenv, "CLAUDE_HISTORY_RAG_EMBEDDING_DIMENSION")
	if err != nil {
		return Config{}, err
	}
	statusPort, err := parseRequiredInt(getenv, "CLAUDE_HISTORY_RAG_STATUS_SERVER_PORT")
	if err != nil {
		return Config{}, err
	}
	config := Config{
		RuntimeContract: strings.TrimSpace(getenv("CLAUDE_HISTORY_RAG_RUNTIME_CONTRACT")),
		Spanner: SpannerTarget{
			Project:  strings.TrimSpace(getenv("CLAUDE_HISTORY_RAG_SPANNER_PROJECT")),
			Instance: strings.TrimSpace(getenv("CLAUDE_HISTORY_RAG_SPANNER_INSTANCE")),
			Database: strings.TrimSpace(getenv("CLAUDE_HISTORY_RAG_SPANNER_DATABASE")),
		},
		StorageBackend:          strings.TrimSpace(getenv("CLAUDE_HISTORY_RAG_STORAGE_BACKEND")),
		SpannerEmbeddingMode:    strings.TrimSpace(getenv("CLAUDE_HISTORY_RAG_SPANNER_EMBEDDING_MODE")),
		SpannerEmbeddingModelID: strings.TrimSpace(getenv("CLAUDE_HISTORY_RAG_SPANNER_EMBEDDING_MODEL_ID")),
		EmbeddingProvider:       strings.TrimSpace(getenv("CLAUDE_HISTORY_RAG_EMBEDDING_PROVIDER")),
		EmbeddingModel:          strings.TrimSpace(getenv("CLAUDE_HISTORY_RAG_EMBEDDING_MODEL")),
		EmbeddingDimension:      embeddingDimension,
		StatusServerHost:        strings.TrimSpace(getenv("CLAUDE_HISTORY_RAG_STATUS_SERVER_HOST")),
		StatusServerPort:        statusPort,
		GoogleCredentials: gcpauth.Selector{
			CredentialsSource:   strings.TrimSpace(getenv("CLAUDE_HISTORY_RAG_CREDENTIALS_SOURCE")),
			CredentialsProfile:  strings.TrimSpace(getenv("CLAUDE_HISTORY_RAG_CREDENTIALS_PROFILE")),
			CredentialsIdentity: strings.TrimSpace(getenv("CLAUDE_HISTORY_RAG_CREDENTIALS_IDENTITY")),
		},
	}
	if err := config.validate(); err != nil {
		return Config{}, err
	}
	return config, nil
}

func (c Config) validate() error {
	for _, expected := range []struct {
		name string
		want string
		got  string
	}{
		{name: "runtime_contract", want: ProductionContract, got: c.RuntimeContract},
		{name: "storage_backend", want: ProductionStorageBackend, got: c.StorageBackend},
		{name: "spanner_embedding_mode", want: ProductionSpannerEmbeddingMode, got: c.SpannerEmbeddingMode},
		{name: "spanner_embedding_model_id", want: ProductionSpannerEmbeddingModelID, got: c.SpannerEmbeddingModelID},
		{name: "embedding_provider", want: ProductionEmbeddingProvider, got: c.EmbeddingProvider},
		{name: "embedding_model", want: ProductionEmbeddingModel, got: c.EmbeddingModel},
		{name: "status_server_host", want: ProductionStatusServerHost, got: c.StatusServerHost},
	} {
		if expected.got != expected.want {
			return fmt.Errorf("history_rag_runtime: %s=%q expected %q", expected.name, expected.got, expected.want)
		}
	}
	for _, required := range []struct {
		name  string
		value string
	}{
		{name: "spanner_project", value: c.Spanner.Project},
		{name: "spanner_instance", value: c.Spanner.Instance},
		{name: "spanner_database", value: c.Spanner.Database},
	} {
		if required.value == "" {
			return fmt.Errorf("history_rag_runtime: %s must be set", required.name)
		}
	}
	if c.EmbeddingDimension != ProductionEmbeddingDimension {
		return fmt.Errorf("history_rag_runtime: embedding_dimension=%d expected %d", c.EmbeddingDimension, ProductionEmbeddingDimension)
	}
	if c.StatusServerPort != ProductionStatusServerPort {
		return fmt.Errorf("history_rag_runtime: status_server_port=%d expected %d", c.StatusServerPort, ProductionStatusServerPort)
	}
	if err := c.GoogleCredentials.Validate("history_rag_google_credentials"); err != nil {
		return err
	}
	return nil
}

func parseRequiredInt(getenv func(string) string, key string) (int, error) {
	raw := strings.TrimSpace(getenv(key))
	if raw == "" {
		return 0, fmt.Errorf("history_rag_runtime: %s is required", strings.ToLower(strings.TrimPrefix(key, "CLAUDE_HISTORY_RAG_")))
	}
	value, err := strconv.Atoi(raw)
	if err != nil {
		return 0, fmt.Errorf("history_rag_runtime: %s must be an integer: %w", strings.ToLower(strings.TrimPrefix(key, "CLAUDE_HISTORY_RAG_")), err)
	}
	return value, nil
}
