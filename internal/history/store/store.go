package store

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"
)

type SpannerStore struct {
	config   Config
	executor Executor

	initializeMu sync.Mutex
	initialized  bool
	closeOnce    sync.Once
	closeErr     error

	stateMu              sync.RWMutex
	vectorIndexAvailable bool
	statsMu              sync.Mutex
	statsCache           *cachedStats

	backfillShard func(context.Context, string) (int64, error)
}

type cachedStats struct {
	at    time.Time
	value Stats
}

var _ Store = (*SpannerStore)(nil)

func New(config Config, executor Executor) (*SpannerStore, error) {
	if err := config.Validate(); err != nil {
		return nil, err
	}
	if executor == nil {
		return nil, fmt.Errorf("executor is required")
	}
	store := &SpannerStore{config: config, executor: executor}
	store.backfillShard = store.runBackfillShard
	return store, nil
}

func (s *SpannerStore) Initialize(ctx context.Context) error {
	if err := contextError(ctx); err != nil {
		return err
	}
	s.initializeMu.Lock()
	defer s.initializeMu.Unlock()
	if s.initialized {
		return nil
	}
	statements, err := BuildInitializationDDL(s.config)
	if err != nil {
		return err
	}
	if err := contextError(ctx); err != nil {
		return err
	}
	if err := s.executor.UpdateDDL(ctx, statements); err != nil {
		return fmt.Errorf("initialize Spanner schema: %w", err)
	}
	s.initialized = true
	return nil
}

func (s *SpannerStore) Upsert(ctx context.Context, chunks []Chunk) error {
	if err := contextError(ctx); err != nil {
		return err
	}
	plan, err := BuildUpsertPlan(s.config, chunks)
	if err != nil {
		return err
	}
	if err := contextError(ctx); err != nil {
		return err
	}
	if plan.Mutation != nil {
		if err := s.executor.Apply(ctx, *plan.Mutation); err != nil {
			return fmt.Errorf("apply chunk mutation: %w", err)
		}
		s.invalidateStats()
		return nil
	}
	if plan.Statement == nil {
		return fmt.Errorf("upsert plan has no operation")
	}
	if err := s.executor.ReadWrite(ctx, func(transaction Transaction) error {
		if err := contextError(ctx); err != nil {
			return err
		}
		affected, err := transaction.Execute(ctx, *plan.Statement)
		if err != nil {
			return err
		}
		if affected < 0 || affected > int64(len(chunks)) {
			return fmt.Errorf("remote-model affected count %d is invalid for batch %d", affected, len(chunks))
		}
		return nil
	}); err != nil {
		return fmt.Errorf("execute remote-model upsert: %w", err)
	}
	s.invalidateStats()
	return nil
}

func (s *SpannerStore) Search(ctx context.Context, query Query) ([]Result, error) {
	if err := contextError(ctx); err != nil {
		return nil, err
	}
	plan, err := BuildSearchPlan(s.config, query, s.vectorIndexReady())
	if err != nil {
		return nil, err
	}
	return s.runSearch(ctx, plan)
}

func (s *SpannerStore) HybridSearch(ctx context.Context, query Query) ([]Result, error) {
	if err := contextError(ctx); err != nil {
		return nil, err
	}
	plan, err := BuildHybridPlan(s.config, query, s.vectorIndexReady())
	if err != nil {
		return nil, err
	}
	return s.runSearch(ctx, plan)
}

func (s *SpannerStore) runSearch(ctx context.Context, plan SearchPlan) ([]Result, error) {
	if err := contextError(ctx); err != nil {
		return nil, err
	}
	rows, err := s.executor.Query(ctx, plan.Statement)
	if err != nil {
		return nil, fmt.Errorf("execute %s search: %w", plan.Type, err)
	}
	results, err := parseResultRows(rows, plan.Type)
	if err != nil {
		return nil, err
	}
	return results, nil
}

func (s *SpannerStore) Stats(ctx context.Context) (Stats, error) {
	if err := contextError(ctx); err != nil {
		return Stats{}, err
	}
	s.statsMu.Lock()
	defer s.statsMu.Unlock()
	if s.statsCache != nil && time.Since(s.statsCache.at) < s.config.StatsCacheTTL {
		return s.statsCache.value, nil
	}
	if err := contextError(ctx); err != nil {
		return Stats{}, err
	}
	statement := Statement{SQL: "SELECT COUNT(*), COUNTIF(Vector IS NOT NULL) FROM ConversationChunks", Params: map[string]any{}}
	rows, err := s.executor.Query(ctx, statement)
	if err != nil {
		return Stats{}, fmt.Errorf("query store stats: %w", err)
	}
	if len(rows) != 1 || len(rows[0]) != 2 {
		return Stats{}, fmt.Errorf("stats query returned an invalid row shape")
	}
	total, ok := toInt64(rows[0][0])
	if !ok || total < 0 {
		return Stats{}, fmt.Errorf("stats total is not a nonnegative integer")
	}
	embedded, ok := toInt64(rows[0][1])
	if !ok || embedded < 0 || embedded > total {
		return Stats{}, fmt.Errorf("stats embedded count is invalid")
	}
	mode := SearchExact
	if s.config.UseANN && s.vectorIndexReady() {
		mode = SearchANN
	}
	stats := Stats{
		TotalChunks: total, EmbeddedChunks: embedded, AwaitingEmbedding: total - embedded,
		Backend: "spanner", Project: s.config.Project, Instance: s.config.Instance,
		Database: s.config.Database, Dimension: VectorDimension,
		FullTextEnabled: s.config.EnableFullText, VectorIndexEnabled: s.vectorIndexReady(),
		VectorSearchMode: mode, EmbeddingStrategy: s.config.EmbeddingStrategy,
		EmbeddingModel: EmbeddingModelName,
	}
	s.statsCache = &cachedStats{at: time.Now(), value: stats}
	return stats, nil
}

func (s *SpannerStore) DeleteMachine(ctx context.Context, machineID string) (int64, error) {
	if err := contextError(ctx); err != nil {
		return 0, err
	}
	machineID = strings.TrimSpace(machineID)
	if machineID == "" || len(machineID) > 256 {
		return 0, fmt.Errorf("machine id must be nonempty and at most 256 bytes")
	}
	statement := Statement{
		SQL:    "DELETE FROM ConversationChunks WHERE MachineId = @machine_id",
		Params: map[string]any{"machine_id": machineID},
	}
	if err := contextError(ctx); err != nil {
		return 0, err
	}
	count, err := s.executor.Execute(ctx, statement)
	if err != nil {
		return 0, fmt.Errorf("delete machine chunks: %w", err)
	}
	if count < 0 {
		return 0, fmt.Errorf("delete machine returned negative affected count")
	}
	s.invalidateStats()
	return count, nil
}

func (s *SpannerStore) Clear(ctx context.Context) (int64, error) {
	if err := contextError(ctx); err != nil {
		return 0, err
	}
	count, err := s.executor.Execute(ctx, Statement{SQL: "DELETE FROM ConversationChunks WHERE TRUE", Params: map[string]any{}})
	if err != nil {
		return 0, fmt.Errorf("clear chunks: %w", err)
	}
	if count < 0 {
		return 0, fmt.Errorf("clear returned negative affected count")
	}
	s.invalidateStats()
	return count, nil
}

func (s *SpannerStore) Optimize(ctx context.Context) error {
	if err := contextError(ctx); err != nil {
		return err
	}
	if !s.config.EnableANN || s.vectorIndexReady() {
		return nil
	}
	rows, err := s.executor.Query(ctx, Statement{SQL: "SELECT COUNT(*) FROM ConversationChunks", Params: map[string]any{}})
	if err != nil {
		return fmt.Errorf("count chunks before vector-index optimization: %w", err)
	}
	if len(rows) != 1 || len(rows[0]) != 1 {
		return fmt.Errorf("optimization count returned an invalid row shape")
	}
	count, ok := toInt64(rows[0][0])
	if !ok || count < 0 {
		return fmt.Errorf("optimization count is invalid")
	}
	if count < VectorIndexThreshold {
		return nil
	}
	ddl, err := BuildVectorIndexDDL(s.config)
	if err != nil {
		return err
	}
	if err := contextError(ctx); err != nil {
		return err
	}
	if err := s.executor.UpdateDDL(ctx, []DDLStatement{ddl}); err != nil {
		return fmt.Errorf("create vector index: %w", err)
	}
	s.stateMu.Lock()
	s.vectorIndexAvailable = true
	s.stateMu.Unlock()
	return nil
}

func (s *SpannerStore) Close() error {
	s.closeOnce.Do(func() { s.closeErr = s.executor.Close() })
	return s.closeErr
}

func (s *SpannerStore) vectorIndexReady() bool {
	s.stateMu.RLock()
	defer s.stateMu.RUnlock()
	return s.vectorIndexAvailable
}

func (s *SpannerStore) invalidateStats() {
	s.statsMu.Lock()
	s.statsCache = nil
	s.statsMu.Unlock()
}

func contextError(ctx context.Context) error {
	if ctx == nil {
		return fmt.Errorf("context is required")
	}
	return ctx.Err()
}

func toInt64(value any) (int64, bool) {
	switch converted := value.(type) {
	case int:
		return int64(converted), true
	case int64:
		return converted, true
	case int32:
		return int64(converted), true
	default:
		return 0, false
	}
}
