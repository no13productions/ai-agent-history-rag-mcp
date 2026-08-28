package store

import (
	"fmt"
	"regexp"
	"strings"
	"time"
)

var identifierPart = regexp.MustCompile(`^[a-z][a-z0-9-]{0,62}[a-z0-9]$|^[a-z]$`)
var locationPart = regexp.MustCompile(`^[a-z]+-[a-z]+[0-9](-[a-z])?$`)

type Config struct {
	Project                    string
	Instance                   string
	Database                   string
	ModelProject               string
	ModelLocation              string
	EmbeddingStrategy          EmbeddingStrategy
	RemoteModel                string
	EmbeddingModel             string
	EmbeddingDimension         int
	DocumentTaskType           string
	QueryTaskType              string
	RemoteRPCBatch             int
	EnableFullText             bool
	EnableANN                  bool
	UseANN                     bool
	VectorIndexLeaves          int
	NumLeavesToSearch          int
	HybridCandidateLimit       int
	RRFK                       int
	MaxSearchLimit             int
	BackfillConcurrency        int
	BackfillBatch              int
	BackfillInterval           time.Duration
	BackfillMaxBatchesPerShard int
	StatsCacheTTL              time.Duration
}

func (c Config) Validate() error {
	for _, field := range []struct {
		name  string
		value string
	}{
		{"project", c.Project},
		{"instance", c.Instance},
		{"database", c.Database},
		{"model project", c.ModelProject},
	} {
		trimmed := strings.TrimSpace(field.value)
		if trimmed == "" || len(trimmed) > 64 || !identifierPart.MatchString(trimmed) {
			return fmt.Errorf("%s must be a nonempty bounded Google resource identifier", field.name)
		}
	}
	if !locationPart.MatchString(c.ModelLocation) {
		return fmt.Errorf("model location must be a valid Google Cloud region")
	}
	if c.EmbeddingStrategy != EmbeddingRemoteModel && c.EmbeddingStrategy != EmbeddingDeferred {
		return fmt.Errorf("embedding strategy must be explicitly %q or %q", EmbeddingRemoteModel, EmbeddingDeferred)
	}
	if c.RemoteModel != RemoteModelName {
		return fmt.Errorf("remote model must be %q", RemoteModelName)
	}
	if c.EmbeddingModel != EmbeddingModelName {
		return fmt.Errorf("embedding model must be %q", EmbeddingModelName)
	}
	if c.EmbeddingDimension != VectorDimension {
		return fmt.Errorf("embedding dimension must be exactly %d", VectorDimension)
	}
	if !validTaskType(c.DocumentTaskType) || !validTaskType(c.QueryTaskType) {
		return fmt.Errorf("embedding task types are invalid")
	}
	if err := bounded("remote RPC batch", c.RemoteRPCBatch, 1, 250); err != nil {
		return err
	}
	if err := bounded("vector index leaves", c.VectorIndexLeaves, 1, 1_000_000); err != nil {
		return err
	}
	if err := bounded("leaves to search", c.NumLeavesToSearch, 1, 1_000_000); err != nil {
		return err
	}
	if err := bounded("hybrid candidate limit", c.HybridCandidateLimit, 1, 10_000); err != nil {
		return err
	}
	if err := bounded("RRF k", c.RRFK, 1, 10_000); err != nil {
		return err
	}
	if err := bounded("maximum search limit", c.MaxSearchLimit, 1, 1_000); err != nil {
		return err
	}
	if err := bounded("backfill concurrency", c.BackfillConcurrency, 1, 256); err != nil {
		return err
	}
	if err := bounded("backfill batch", c.BackfillBatch, 1, 2_000); err != nil {
		return err
	}
	if c.BackfillInterval < 10*time.Second || c.BackfillInterval > time.Hour {
		return fmt.Errorf("backfill interval must be between 10 seconds and 1 hour")
	}
	if err := bounded("backfill batches per shard", c.BackfillMaxBatchesPerShard, 1, 1_000_000); err != nil {
		return err
	}
	if c.StatsCacheTTL < time.Second || c.StatsCacheTTL > time.Minute {
		return fmt.Errorf("stats cache TTL must be between 1 second and 1 minute")
	}
	if c.UseANN && !c.EnableANN {
		return fmt.Errorf("ANN use requires ANN to be enabled")
	}
	return nil
}

func validTaskType(value string) bool {
	switch value {
	case TaskRetrievalQuery, TaskRetrievalDocument, TaskSemanticSimilarity:
		return true
	default:
		return false
	}
}

func bounded(name string, value, minimum, maximum int) error {
	if value < minimum || value > maximum {
		return fmt.Errorf("%s must be in [%d,%d]", name, minimum, maximum)
	}
	return nil
}
