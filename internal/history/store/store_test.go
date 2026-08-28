package store

import (
	"context"
	"errors"
	"math"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"
)

type fakeExecutor struct {
	mu          sync.Mutex
	queries     []Statement
	executions  []Statement
	mutations   []Mutation
	ddl         [][]DDLStatement
	queryRows   []Row
	queryErr    error
	queryQueue  []queryResponse
	executeRows int64
	executeErr  error
	closeCalls  int
}

type queryResponse struct {
	rows []Row
	err  error
}

func (f *fakeExecutor) Query(_ context.Context, statement Statement) ([]Row, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.queries = append(f.queries, statement)
	if len(f.queryQueue) != 0 {
		response := f.queryQueue[0]
		f.queryQueue = f.queryQueue[1:]
		return response.rows, response.err
	}
	return f.queryRows, f.queryErr
}

func (f *fakeExecutor) Execute(_ context.Context, statement Statement) (int64, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.executions = append(f.executions, statement)
	return f.executeRows, f.executeErr
}

func (f *fakeExecutor) Apply(_ context.Context, mutation Mutation) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.mutations = append(f.mutations, mutation)
	return nil
}

func (f *fakeExecutor) ReadWrite(ctx context.Context, operation func(Transaction) error) error {
	return operation(f)
}

func (f *fakeExecutor) UpdateDDL(_ context.Context, statements []DDLStatement) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.ddl = append(f.ddl, append([]DDLStatement(nil), statements...))
	return nil
}

func (f *fakeExecutor) Close() error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.closeCalls++
	return nil
}

func validConfig(strategy EmbeddingStrategy) Config {
	return Config{
		Project:                    "project",
		Instance:                   "instance",
		Database:                   "database",
		ModelProject:               "model-project",
		ModelLocation:              "us-central1",
		EmbeddingStrategy:          strategy,
		RemoteModel:                RemoteModelName,
		EmbeddingModel:             EmbeddingModelName,
		EmbeddingDimension:         VectorDimension,
		DocumentTaskType:           TaskRetrievalDocument,
		QueryTaskType:              TaskRetrievalQuery,
		RemoteRPCBatch:             1,
		EnableFullText:             true,
		EnableANN:                  true,
		UseANN:                     true,
		VectorIndexLeaves:          1000,
		NumLeavesToSearch:          50,
		HybridCandidateLimit:       100,
		RRFK:                       60,
		MaxSearchLimit:             100,
		BackfillConcurrency:        8,
		BackfillBatch:              200,
		BackfillInterval:           time.Minute,
		BackfillMaxBatchesPerShard: 1000,
		StatsCacheTTL:              10 * time.Second,
	}
}

func validChunk(id string) Chunk {
	return Chunk{
		ID:          id,
		Content:     "content",
		ChunkType:   "turn",
		SessionID:   "session",
		ProjectPath: "/project",
		ProjectName: "project",
		Timestamp:   time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC),
		SourceFile:  "/source.jsonl",
		SourceLine:  1,
		MachineID:   "machine",
	}
}

func validVector() []float32 {
	vector := make([]float32, VectorDimension)
	vector[0] = 1
	return vector
}

func validVector64() []float64 {
	vector := make([]float64, VectorDimension)
	vector[0] = 1
	return vector
}

func validResultRow() Row {
	return Row{
		"id", "content", "turn", "session", "/project", "project",
		time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC), nil, nil, nil, float64(0.25),
	}
}

func TestConfigRequiresExplicitProductionEmbeddingStrategy(t *testing.T) {
	config := validConfig("")
	if err := config.Validate(); err == nil || !strings.Contains(err.Error(), "embedding strategy") {
		t.Fatalf("Validate() error = %v, want explicit strategy refusal", err)
	}
	for _, bad := range []Config{
		func() Config { c := validConfig(EmbeddingRemoteModel); c.RemoteModel = "bad"; return c }(),
		func() Config { c := validConfig(EmbeddingRemoteModel); c.EmbeddingModel = "bad"; return c }(),
		func() Config { c := validConfig(EmbeddingRemoteModel); c.EmbeddingDimension = 768; return c }(),
		func() Config { c := validConfig(EmbeddingRemoteModel); c.RemoteRPCBatch = 2; return c }(),
		func() Config { c := validConfig(EmbeddingRemoteModel); c.BackfillConcurrency = 257; return c }(),
		func() Config { c := validConfig(EmbeddingRemoteModel); c.BackfillBatch = 2001; return c }(),
		func() Config { c := validConfig(EmbeddingRemoteModel); c.BackfillInterval = 9 * time.Second; return c }(),
	} {
		if err := bad.Validate(); err == nil {
			t.Fatalf("Validate() accepted invalid config: %+v", bad)
		}
	}
}

func TestConfigClosesEmbeddingRolesRegionAndBatch(t *testing.T) {
	for name, mutate := range map[string]func(*Config){
		"document role": func(config *Config) { config.DocumentTaskType = TaskRetrievalQuery },
		"query role":    func(config *Config) { config.QueryTaskType = TaskRetrievalDocument },
		"zone":          func(config *Config) { config.ModelLocation = "us-central1-a" },
		"batch zero":    func(config *Config) { config.RemoteRPCBatch = 0 },
		"batch two":     func(config *Config) { config.RemoteRPCBatch = 2 },
	} {
		t.Run(name, func(t *testing.T) {
			config := validConfig(EmbeddingRemoteModel)
			mutate(&config)
			if err := config.Validate(); err == nil {
				t.Fatalf("Validate() accepted %+v", config)
			}
		})
	}
}

func TestSchemaPlansPinCanonicalObjects(t *testing.T) {
	plans, err := BuildInitializationDDL(validConfig(EmbeddingRemoteModel))
	if err != nil {
		t.Fatal(err)
	}
	var sql []string
	for _, plan := range plans {
		if !plan.AlreadyExistsOK {
			t.Fatalf("initialization DDL is not idempotent: %#v", plan)
		}
		sql = append(sql, plan.SQL)
	}
	joined := strings.Join(sql, "\n")
	for _, required := range []string{
		"CREATE TABLE ConversationChunks",
		"Vector ARRAY<FLOAT32>(vector_length=>3072)",
		"ContentTokens TOKENLIST AS (TOKENIZE_FULLTEXT(Content)) HIDDEN",
		"CREATE SEARCH INDEX ConversationChunksContentSearch",
		"CREATE MODEL IF NOT EXISTS ConversationEmbeddingModel",
		"gemini-embedding-001",
	} {
		if !strings.Contains(joined, required) {
			t.Fatalf("initialization DDL missing %q:\n%s", required, joined)
		}
	}
	vectorDDL, err := BuildVectorIndexDDL(validConfig(EmbeddingRemoteModel))
	if err != nil {
		t.Fatal(err)
	}
	if !vectorDDL.AlreadyExistsOK || !strings.Contains(vectorDDL.SQL, "WHERE Vector IS NOT NULL") {
		t.Fatalf("vector DDL must exclude NULL vectors: %s", vectorDDL.SQL)
	}
}

func TestInitializeIsSingleFlight(t *testing.T) {
	executor := &fakeExecutor{}
	store, err := New(validConfig(EmbeddingRemoteModel), executor)
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	for range 16 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if initializeErr := store.Initialize(context.Background()); initializeErr != nil {
				t.Errorf("Initialize() error = %v", initializeErr)
			}
		}()
	}
	wg.Wait()
	if got := len(executor.ddl); got != 1 {
		t.Fatalf("UpdateDDL calls = %d, want 1", got)
	}
}

func TestUpsertRejectsMixedBatchBeforeExecutor(t *testing.T) {
	executor := &fakeExecutor{}
	store, _ := New(validConfig(EmbeddingRemoteModel), executor)
	embedded := validChunk("embedded")
	embedded.Vector = validVector()
	if err := store.Upsert(context.Background(), []Chunk{validChunk("raw"), embedded}); err == nil {
		t.Fatal("Upsert() accepted mixed embedded/unembedded batch")
	}
	if len(executor.executions)+len(executor.mutations) != 0 {
		t.Fatal("executor called for rejected mixed batch")
	}
}

func TestUpsertValidatesVectorDimensionFiniteAndNonzero(t *testing.T) {
	for name, vector := range map[string][]float32{
		"dimension": {1},
		"nan":       func() []float32 { v := validVector(); v[4] = float32(math.NaN()); return v }(),
		"infinity":  func() []float32 { v := validVector(); v[4] = float32(math.Inf(1)); return v }(),
		"zero":      make([]float32, VectorDimension),
	} {
		t.Run(name, func(t *testing.T) {
			executor := &fakeExecutor{}
			store, _ := New(validConfig(EmbeddingRemoteModel), executor)
			chunk := validChunk("bad")
			chunk.Vector = vector
			if err := store.Upsert(context.Background(), []Chunk{chunk}); err == nil {
				t.Fatal("Upsert() accepted invalid vector")
			}
			if len(executor.mutations) != 0 {
				t.Fatal("executor called for invalid vector")
			}
		})
	}
}

func TestUpsertPlansAreDeterministicAndParameterized(t *testing.T) {
	remote := validConfig(EmbeddingRemoteModel)
	first := validChunk("b")
	first.Content = "chunk content'); DELETE FROM ConversationChunks WHERE TRUE; --"
	planA, err := BuildUpsertPlan(remote, []Chunk{first, validChunk("a")})
	if err != nil {
		t.Fatal(err)
	}
	planB, err := BuildUpsertPlan(remote, []Chunk{first, validChunk("a")})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(planA, planB) {
		t.Fatal("remote upsert plan is not deterministic")
	}
	if planA.Statement == nil || !strings.Contains(planA.Statement.SQL, "ML.PREDICT") {
		t.Fatalf("remote plan = %#v, want ML.PREDICT statement", planA)
	}
	if strings.Contains(planA.Statement.SQL, first.Content) {
		t.Fatal("caller content interpolated into SQL")
	}
	rows, ok := planA.Statement.Params["rows"].([]RemoteEmbeddingRow)
	if !ok || len(rows) != 2 || rows[0].Content != first.Content {
		t.Fatal("remote plan lacks bound rows")
	}

	deferred := validConfig(EmbeddingDeferred)
	plan, err := BuildUpsertPlan(deferred, []Chunk{validChunk("abraw")})
	if err != nil {
		t.Fatal(err)
	}
	if plan.Mutation == nil || contains(plan.Mutation.Columns, "Vector") {
		t.Fatalf("deferred mutation must omit Vector: %#v", plan.Mutation)
	}

	embedded := validChunk("embedded")
	embedded.Vector = validVector()
	plan, err = BuildUpsertPlan(remote, []Chunk{embedded})
	if err != nil {
		t.Fatal(err)
	}
	if plan.Mutation == nil || !reflect.DeepEqual(plan.Mutation.Columns, allColumns) {
		t.Fatalf("embedded mutation columns = %#v, want %#v", plan.Mutation, allColumns)
	}
	values := plan.Mutation.Values[0]
	if values[0] != "embedded" || !reflect.DeepEqual(values[2], embedded.Vector) || values[17] != "machine" {
		t.Fatalf("embedded mutation field order is wrong: %#v", values)
	}
}

func TestDeferredUpsertRequiresReachableHexShardPrefix(t *testing.T) {
	config := validConfig(EmbeddingDeferred)
	if _, err := BuildUpsertPlan(config, []Chunk{validChunk("chunk-not-hex")}); err == nil {
		t.Fatal("deferred upsert accepted an unreachable id")
	}
	if _, err := BuildUpsertPlan(config, []Chunk{validChunk("0achunk")}); err != nil {
		t.Fatalf("reachable deferred id refused: %v", err)
	}
}

func TestSearchPlansAlwaysExcludeNullVectorsAndBindValues(t *testing.T) {
	config := validConfig(EmbeddingRemoteModel)
	query := Query{
		Vector: validVector(), Limit: 7, Mode: SearchExact,
		Filter: Filter{ProjectPath: "x' OR TRUE --", ChunkType: "turn"},
	}
	plan, err := BuildSearchPlan(config, query, true)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(plan.Statement.SQL, "Vector IS NOT NULL") {
		t.Fatalf("search SQL lost NULL-vector exclusion: %s", plan.Statement.SQL)
	}
	if strings.Contains(plan.Statement.SQL, query.Filter.ProjectPath) {
		t.Fatal("caller filter interpolated into SQL")
	}
	if plan.Statement.Params["project_path"] != query.Filter.ProjectPath {
		t.Fatalf("project filter not bound: %#v", plan.Statement.Params)
	}
	if !strings.Contains(plan.Statement.SQL, "LIMIT 7") || strings.Contains(plan.Statement.SQL, "LIMIT @") {
		t.Fatalf("search limit is not a validated literal: %s", plan.Statement.SQL)
	}
}

func TestANNSelectionRefusesUnstoredFilters(t *testing.T) {
	config := validConfig(EmbeddingRemoteModel)
	query := Query{Vector: validVector(), Limit: 5, Mode: SearchANN, Filter: Filter{ProjectPath: "/p"}}
	if _, err := BuildSearchPlan(config, query, true); err == nil || !strings.Contains(err.Error(), "ANN") {
		t.Fatalf("BuildSearchPlan() error = %v, want ANN filter refusal", err)
	}
	query.Mode = SearchAuto
	plan, err := BuildSearchPlan(config, query, true)
	if err != nil {
		t.Fatal(err)
	}
	if plan.Mode != SearchExact || strings.Contains(plan.Statement.SQL, "FORCE_INDEX") {
		t.Fatalf("auto mode did not lawfully select exact search: %#v", plan)
	}
	query.Filter = Filter{ChunkType: "turn", SessionID: "s", ProjectName: "p", MachineID: "m"}
	plan, err = BuildSearchPlan(config, query, true)
	if err != nil {
		t.Fatal(err)
	}
	if plan.Mode != SearchANN || !strings.Contains(plan.Statement.SQL, "FORCE_INDEX=ConversationChunksVectorIndex") {
		t.Fatalf("covered ANN plan = %#v", plan)
	}
}

func TestSearchLimitRejectedBeforeExecutor(t *testing.T) {
	executor := &fakeExecutor{}
	store, _ := New(validConfig(EmbeddingRemoteModel), executor)
	for _, limit := range []int{0, 101} {
		_, err := store.Search(context.Background(), Query{Vector: validVector(), Limit: limit, Mode: SearchExact})
		if err == nil {
			t.Fatalf("Search(limit=%d) unexpectedly succeeded", limit)
		}
	}
	if len(executor.queries) != 0 {
		t.Fatal("executor called for invalid search limit")
	}
}

func TestSearchEmbedsQueryThroughSpannerQueryTask(t *testing.T) {
	executor := &fakeExecutor{queryQueue: []queryResponse{
		{rows: []Row{{validVector64()}}},
		{rows: []Row{validResultRow()}},
	}}
	store, err := New(validConfig(EmbeddingRemoteModel), executor)
	if err != nil {
		t.Fatal(err)
	}
	results, err := store.Search(context.Background(), Query{Text: "oauth", Limit: 1, Mode: SearchExact})
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 1 || results[0].SearchType != SearchTypeExact {
		t.Fatalf("Search() = %#v", results)
	}
	if len(executor.queries) != 2 || !strings.Contains(executor.queries[0].SQL, "ML.PREDICT") {
		t.Fatalf("query embedding path = %#v", executor.queries)
	}
	if executor.queries[0].Params["task_type"] != TaskRetrievalQuery {
		t.Fatalf("query task type = %#v", executor.queries[0].Params)
	}
	if !strings.Contains(executor.queries[0].SQL, "remote_udf_max_rows_per_rpc=1") {
		t.Fatalf("query embedding batch is not closed: %s", executor.queries[0].SQL)
	}
}

func TestHybridPlanUsesFTSAndRRFWithoutLosingVectorGuard(t *testing.T) {
	query := Query{Text: "oauth", Vector: validVector(), Limit: 5, Mode: SearchAuto}
	plan, err := BuildHybridPlan(validConfig(EmbeddingRemoteModel), query, true)
	if err != nil {
		t.Fatal(err)
	}
	for _, fragment := range []string{"SEARCH(ContentTokens, @query)", "SCORE(ContentTokens, @query)", "Vector IS NOT NULL", "SUM(1.0 / (@rrf_k + rank + 1))", "LIMIT 100", "LIMIT 5"} {
		if !strings.Contains(plan.Statement.SQL, fragment) {
			t.Fatalf("hybrid SQL missing %q:\n%s", fragment, plan.Statement.SQL)
		}
	}
}

func TestHybridSearchFallsBackTruthfullyToVector(t *testing.T) {
	executor := &fakeExecutor{queryQueue: []queryResponse{
		{err: errors.New("full-text index unavailable")},
		{rows: []Row{validResultRow()}},
	}}
	store, err := New(validConfig(EmbeddingRemoteModel), executor)
	if err != nil {
		t.Fatal(err)
	}
	results, err := store.HybridSearch(context.Background(), Query{
		Text: "oauth", Vector: validVector(), Limit: 1, Mode: SearchExact,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 1 || results[0].SearchType != SearchTypeExact {
		t.Fatalf("fallback results = %#v", results)
	}
	if len(executor.queries) != 2 || !strings.Contains(executor.queries[0].SQL, "SEARCH(") || strings.Contains(executor.queries[1].SQL, "SEARCH(") {
		t.Fatalf("hybrid fallback queries = %#v", executor.queries)
	}
}

func TestResultRowsRejectMalformedFieldsAndNonfiniteDistance(t *testing.T) {
	for name, mutate := range map[string]func(Row){
		"chunk type": func(row Row) { row[2] = int64(1) },
		"timestamp":  func(row Row) { row[6] = time.Time{} },
		"optional":   func(row Row) { row[7] = int64(1) },
		"nan":        func(row Row) { row[10] = math.NaN() },
		"infinity":   func(row Row) { row[10] = math.Inf(1) },
	} {
		t.Run(name, func(t *testing.T) {
			row := validResultRow()
			mutate(row)
			if _, err := parseResultRows([]Row{row}, SearchTypeExact); err == nil {
				t.Fatalf("parseResultRows accepted %#v", row)
			}
		})
	}
}

func TestDeleteMachineRetainsPredicateAndTypedCount(t *testing.T) {
	executor := &fakeExecutor{executeRows: 7}
	store, _ := New(validConfig(EmbeddingRemoteModel), executor)
	count, err := store.DeleteMachine(context.Background(), "machine' OR TRUE --")
	if err != nil {
		t.Fatal(err)
	}
	if count != 7 {
		t.Fatalf("DeleteMachine() count = %d, want 7", count)
	}
	statement := executor.executions[0]
	if !strings.Contains(statement.SQL, "WHERE MachineId = @machine_id") {
		t.Fatalf("machine predicate missing: %s", statement.SQL)
	}
	if strings.Contains(statement.SQL, "machine' OR") {
		t.Fatal("machine id interpolated into SQL")
	}
}

func TestClearReturnsExecutorAffectedCount(t *testing.T) {
	executor := &fakeExecutor{executeRows: 9}
	store, _ := New(validConfig(EmbeddingRemoteModel), executor)
	count, err := store.Clear(context.Background())
	if err != nil || count != 9 {
		t.Fatalf("Clear() = (%d, %v), want (9, nil)", count, err)
	}
}

func TestCancellationStopsBeforeExecutor(t *testing.T) {
	executor := &fakeExecutor{}
	store, _ := New(validConfig(EmbeddingRemoteModel), executor)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	chunk := validChunk("embedded")
	chunk.Vector = validVector()
	if err := store.Upsert(ctx, []Chunk{chunk}); !errors.Is(err, context.Canceled) {
		t.Fatalf("Upsert() error = %v, want context.Canceled", err)
	}
	if _, err := store.Search(ctx, Query{Vector: validVector(), Limit: 1, Mode: SearchExact}); !errors.Is(err, context.Canceled) {
		t.Fatalf("Search() error = %v, want context.Canceled", err)
	}
	if len(executor.queries)+len(executor.executions)+len(executor.mutations) != 0 {
		t.Fatal("executor called after context cancellation")
	}
}

func TestBackfillShardsAreExactDeterministicHexSpace(t *testing.T) {
	first := BackfillShards()
	second := BackfillShards()
	if len(first) != 256 || !reflect.DeepEqual(first, second) {
		t.Fatalf("BackfillShards() length/determinism failure: %d", len(first))
	}
	seen := make(map[string]bool, len(first))
	for index, prefix := range first {
		want := strings.ToLower(hexByte(byte(index)))
		if prefix != want || seen[prefix] {
			t.Fatalf("shard %d = %q, want unique %q", index, prefix, want)
		}
		seen[prefix] = true
	}
}

func TestBackfillReadPlanIsBoundedAndNullOnly(t *testing.T) {
	plan, err := BuildBackfillReadPlan(validConfig(EmbeddingDeferred), "ab")
	if err != nil {
		t.Fatal(err)
	}
	for _, fragment := range []string{"Vector IS NULL", "STARTS_WITH(Id, @prefix)", "LIMIT 200"} {
		if !strings.Contains(plan.SQL, fragment) {
			t.Fatalf("backfill SQL missing %q: %s", fragment, plan.SQL)
		}
	}
	if plan.Params["prefix"] != "ab" {
		t.Fatalf("prefix not bound: %#v", plan.Params)
	}
}

func TestBackfillIsolatesShardFailureAndPreservesNullRetry(t *testing.T) {
	executor := &fakeExecutor{}
	store, _ := New(validConfig(EmbeddingDeferred), executor)
	store.backfillShard = func(_ context.Context, prefix string) (int64, error) {
		if prefix == "7f" {
			return 0, errors.New("quota")
		}
		return 1, nil
	}
	report, err := store.Backfill(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if report.Embedded != 255 || report.Failures["7f"] == "" {
		t.Fatalf("Backfill() report = %#v", report)
	}
	if len(executor.mutations)+len(executor.executions) != 0 {
		t.Fatal("test shard failure unexpectedly mutated rows; failed rows must remain NULL")
	}
}

func TestStatsCalculatesTypedAwaitingCount(t *testing.T) {
	executor := &fakeExecutor{queryRows: []Row{{int64(10), int64(7)}}}
	store, _ := New(validConfig(EmbeddingRemoteModel), executor)
	stats, err := store.Stats(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if stats.TotalChunks != 10 || stats.EmbeddedChunks != 7 || stats.AwaitingEmbedding != 3 {
		t.Fatalf("Stats() = %#v", stats)
	}
}

func TestStatsReportsBackfillRateAndETA(t *testing.T) {
	config := validConfig(EmbeddingRemoteModel)
	config.StatsCacheTTL = time.Second
	executor := &fakeExecutor{queryQueue: []queryResponse{
		{rows: []Row{{int64(1000), int64(700)}}},
		{rows: []Row{{int64(1000), int64(800)}}},
	}}
	store, err := New(config, executor)
	if err != nil {
		t.Fatal(err)
	}
	now := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	store.now = func() time.Time { return now }
	if _, err := store.Stats(context.Background()); err != nil {
		t.Fatal(err)
	}
	now = now.Add(30 * time.Second)
	stats, err := store.Stats(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if stats.BackfillRatePerMinute != 200 || stats.BackfillETASeconds == nil || *stats.BackfillETASeconds != 60 {
		t.Fatalf("backfill status = %#v", stats)
	}
}

func TestChunkExistsIsBoundAndTyped(t *testing.T) {
	executor := &fakeExecutor{queryRows: []Row{{true}}}
	store, err := New(validConfig(EmbeddingRemoteModel), executor)
	if err != nil {
		t.Fatal(err)
	}
	exists, err := store.ChunkExists(context.Background(), "chunk-id")
	if err != nil || !exists {
		t.Fatalf("ChunkExists() = (%v, %v)", exists, err)
	}
	if len(executor.queries) != 1 || executor.queries[0].Params["id"] != "chunk-id" ||
		!strings.Contains(executor.queries[0].SQL, "WHERE Id = @id") || strings.Contains(executor.queries[0].SQL, "chunk-id") {
		t.Fatalf("chunk existence query = %#v", executor.queries)
	}
	if _, err := store.ChunkExists(context.Background(), " "); err == nil {
		t.Fatal("ChunkExists accepted empty id")
	}
	if _, err := store.ChunkExists(context.Background(), " chunk-id"); err == nil {
		t.Fatal("ChunkExists normalized a noncanonical id")
	}
}

func TestStatsIsSingleFlightAndCached(t *testing.T) {
	executor := &fakeExecutor{queryRows: []Row{{int64(10), int64(7)}}}
	store, _ := New(validConfig(EmbeddingRemoteModel), executor)
	var wg sync.WaitGroup
	for range 16 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, err := store.Stats(context.Background()); err != nil {
				t.Errorf("Stats() error = %v", err)
			}
		}()
	}
	wg.Wait()
	if got := len(executor.queries); got != 1 {
		t.Fatalf("stats queries = %d, want 1", got)
	}
}

func TestOptimizeCreatesDeferredANNOnlyAfterThreshold(t *testing.T) {
	executor := &fakeExecutor{queryRows: []Row{{int64(VectorIndexThreshold)}}}
	store, _ := New(validConfig(EmbeddingRemoteModel), executor)
	if err := store.Optimize(context.Background()); err != nil {
		t.Fatal(err)
	}
	if len(executor.ddl) != 1 || !strings.Contains(executor.ddl[0][0].SQL, "CREATE VECTOR INDEX") {
		t.Fatalf("Optimize() DDL calls = %#v", executor.ddl)
	}
}

func TestCloseIsIdempotent(t *testing.T) {
	executor := &fakeExecutor{}
	store, _ := New(validConfig(EmbeddingRemoteModel), executor)
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	if executor.closeCalls != 1 {
		t.Fatalf("Close() executor calls = %d, want 1", executor.closeCalls)
	}
}

func contains(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}
