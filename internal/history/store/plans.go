package store

import (
	"fmt"
	"math"
	"strings"
	"time"
)

var allColumns = []string{
	"Id", "Content", "Vector", "ChunkType", "SessionId", "ProjectPath",
	"ProjectName", "Timestamp", "UserUuid", "AssistantUuid", "FilePath",
	"Operation", "Model", "SourceFile", "SourceLine", "ParentChunkId",
	"ChildChunkIds", "MachineId",
}

var columnsWithoutVector = []string{
	"Id", "Content", "ChunkType", "SessionId", "ProjectPath", "ProjectName",
	"Timestamp", "UserUuid", "AssistantUuid", "FilePath", "Operation", "Model",
	"SourceFile", "SourceLine", "ParentChunkId", "ChildChunkIds", "MachineId",
}

const maxQueryTextBytes = 16_384

func BuildInitializationDDL(config Config) ([]DDLStatement, error) {
	if err := config.Validate(); err != nil {
		return nil, err
	}
	statements := []DDLStatement{
		{SQL: `CREATE TABLE ConversationChunks (
  Id STRING(64) NOT NULL,
  Content STRING(MAX) NOT NULL,
  ContentTokens TOKENLIST AS (TOKENIZE_FULLTEXT(Content)) HIDDEN,
  Vector ARRAY<FLOAT32>(vector_length=>3072),
  ChunkType STRING(32) NOT NULL,
  SessionId STRING(128) NOT NULL,
  ProjectPath STRING(MAX) NOT NULL,
  ProjectName STRING(256) NOT NULL,
  Timestamp TIMESTAMP NOT NULL,
  UserUuid STRING(128),
  AssistantUuid STRING(128),
  FilePath STRING(MAX),
  Operation STRING(32),
  Model STRING(256),
  SourceFile STRING(MAX) NOT NULL,
  SourceLine INT64 NOT NULL,
  ParentChunkId STRING(64),
  ChildChunkIds ARRAY<STRING(64)>,
  MachineId STRING(256)
) PRIMARY KEY (Id)`, AlreadyExistsOK: true},
		{SQL: `CREATE INDEX ConversationChunksByProject ON ConversationChunks(ProjectPath)`, AlreadyExistsOK: true},
		{SQL: `CREATE INDEX ConversationChunksByMachine ON ConversationChunks(MachineId)`, AlreadyExistsOK: true},
		{SQL: `CREATE INDEX ConversationChunksByTimestamp ON ConversationChunks(Timestamp)`, AlreadyExistsOK: true},
		{SQL: `CREATE INDEX ConversationChunksByProjectTimestamp ON ConversationChunks(ProjectPath, Timestamp)`, AlreadyExistsOK: true},
	}
	if config.EnableFullText {
		statements = append(statements, DDLStatement{SQL: `CREATE SEARCH INDEX ConversationChunksContentSearch ON ConversationChunks(ContentTokens)`, AlreadyExistsOK: true})
	}
	endpoint := fmt.Sprintf("//aiplatform.googleapis.com/projects/%s/locations/%s/publishers/google/models/%s", config.ModelProject, config.ModelLocation, EmbeddingModelName)
	statements = append(statements, DDLStatement{SQL: fmt.Sprintf(`CREATE MODEL IF NOT EXISTS %s
INPUT(content STRING(MAX), task_type STRING(MAX))
OUTPUT(embeddings STRUCT<statistics STRUCT<truncated BOOL, token_count FLOAT64>, values ARRAY<FLOAT64>>)
REMOTE OPTIONS (endpoint = '%s', default_batch_size = %d)`, RemoteModelName, endpoint, config.RemoteRPCBatch), AlreadyExistsOK: true})
	return statements, nil
}

func BuildVectorIndexDDL(config Config) (DDLStatement, error) {
	if err := config.Validate(); err != nil {
		return DDLStatement{}, err
	}
	return DDLStatement{SQL: fmt.Sprintf(`CREATE VECTOR INDEX ConversationChunksVectorIndex
ON ConversationChunks(Vector)
STORING (ChunkType, SessionId, ProjectName, MachineId)
WHERE Vector IS NOT NULL
OPTIONS (distance_type = 'COSINE', tree_depth = 2, num_leaves = %d)`, config.VectorIndexLeaves), AlreadyExistsOK: true}, nil
}

func BuildUpsertPlan(config Config, chunks []Chunk) (UpsertPlan, error) {
	if err := config.Validate(); err != nil {
		return UpsertPlan{}, err
	}
	if len(chunks) == 0 {
		return UpsertPlan{}, fmt.Errorf("upsert batch must not be empty")
	}
	if len(chunks) > 500 {
		return UpsertPlan{}, fmt.Errorf("upsert batch exceeds 500 chunks")
	}
	embedded, unembedded := 0, 0
	for index := range chunks {
		if err := validateChunk(chunks[index]); err != nil {
			return UpsertPlan{}, fmt.Errorf("chunk %d: %w", index, err)
		}
		if config.EmbeddingStrategy == EmbeddingDeferred && len(chunks[index].Vector) == 0 && !hasLowerHexShardPrefix(chunks[index].ID) {
			return UpsertPlan{}, fmt.Errorf("chunk %d: deferred id must begin with two lowercase hexadecimal characters", index)
		}
		if len(chunks[index].Vector) == 0 {
			unembedded++
		} else {
			embedded++
			if err := validateVector(chunks[index].Vector); err != nil {
				return UpsertPlan{}, fmt.Errorf("chunk %q: %w", chunks[index].ID, err)
			}
		}
	}
	if embedded != 0 && unembedded != 0 {
		return UpsertPlan{}, fmt.Errorf("cannot mix embedded and unembedded chunks in one batch")
	}
	if embedded != 0 {
		mutation := Mutation{Table: TableName, Columns: append([]string(nil), allColumns...)}
		for _, chunk := range chunks {
			mutation.Values = append(mutation.Values, chunkValues(chunk, true))
		}
		return UpsertPlan{Mutation: &mutation}, nil
	}
	if config.EmbeddingStrategy == EmbeddingDeferred {
		mutation := Mutation{Table: TableName, Columns: append([]string(nil), columnsWithoutVector...)}
		for _, chunk := range chunks {
			mutation.Values = append(mutation.Values, chunkValues(chunk, false))
		}
		return UpsertPlan{Mutation: &mutation}, nil
	}
	statement := buildRemoteModelStatement(config, chunks)
	return UpsertPlan{Statement: &statement}, nil
}

func BuildQueryEmbeddingPlan(config Config, text string) (Statement, error) {
	if err := config.Validate(); err != nil {
		return Statement{}, err
	}
	if err := validateQueryText(text); err != nil {
		return Statement{}, err
	}
	return Statement{
		SQL: `SELECT ARRAY(
  SELECT CAST(value AS FLOAT32) FROM UNNEST(embeddings.values) AS value
) AS Vector
FROM ML.PREDICT(
  MODEL ConversationEmbeddingModel,
  (SELECT @content AS content, @task_type AS task_type),
  STRUCT(3072 AS outputDimensionality)
) @{remote_udf_max_rows_per_rpc=1}`,
		Params: map[string]any{"content": text, "task_type": config.QueryTaskType},
	}, nil
}

func buildRemoteModelStatement(config Config, chunks []Chunk) Statement {
	rows := make([]RemoteEmbeddingRow, 0, len(chunks))
	for _, chunk := range chunks {
		rows = append(rows, chunkRow(chunk))
	}
	return Statement{
		SQL: fmt.Sprintf(`INSERT OR UPDATE INTO ConversationChunks (
  Id, Content, Vector, ChunkType, SessionId, ProjectPath, ProjectName, Timestamp,
  UserUuid, AssistantUuid, FilePath, Operation, Model, SourceFile, SourceLine,
  ParentChunkId, ChildChunkIds, MachineId
)
SELECT pred.id, pred.content,
  ARRAY(SELECT CAST(value AS FLOAT32) FROM UNNEST(pred.embeddings.values) AS value),
  pred.chunk_type, pred.session_id, pred.project_path, pred.project_name, pred.timestamp,
  pred.user_uuid, pred.assistant_uuid, pred.file_path, pred.operation, pred.model,
  pred.source_file, pred.source_line, pred.parent_chunk_id, pred.child_chunk_ids,
  pred.machine_id
FROM ML.PREDICT(
  MODEL ConversationEmbeddingModel,
  (SELECT r.*, @task_type AS task_type FROM UNNEST(@rows) AS r),
  STRUCT(3072 AS outputDimensionality)
) @{remote_udf_max_rows_per_rpc=%d} AS pred`, config.RemoteRPCBatch),
		Params: map[string]any{"rows": rows, "task_type": config.DocumentTaskType},
	}
}

func BuildSearchPlan(config Config, query Query, vectorIndexAvailable bool) (SearchPlan, error) {
	if err := config.Validate(); err != nil {
		return SearchPlan{}, err
	}
	if err := validateLimit(query.Limit, config.MaxSearchLimit); err != nil {
		return SearchPlan{}, err
	}
	if err := validateVector(query.Vector); err != nil {
		return SearchPlan{}, fmt.Errorf("query vector: %w", err)
	}
	mode, err := chooseVectorMode(config, query, vectorIndexAvailable)
	if err != nil {
		return SearchPlan{}, err
	}
	filters, params, err := buildFilters(query.Filter, true)
	if err != nil {
		return SearchPlan{}, err
	}
	params["query_vector"] = append([]float32(nil), query.Vector...)
	table := TableName
	distance := "COSINE_DISTANCE(Vector, @query_vector)"
	searchType := SearchTypeExact
	if mode == SearchANN {
		table = TableName + "@{FORCE_INDEX=" + VectorIndexName + "}"
		distance = fmt.Sprintf(`APPROX_COSINE_DISTANCE(Vector, @query_vector, options => JSON '{"num_leaves_to_search": %d}')`, config.NumLeavesToSearch)
		searchType = SearchTypeANN
	}
	sql := fmt.Sprintf(`SELECT Id, Content, ChunkType, SessionId, ProjectPath, ProjectName,
  Timestamp, FilePath, Operation, MachineId, %s AS Distance
FROM %s
WHERE %s
ORDER BY Distance ASC
LIMIT %d`, distance, table, strings.Join(filters, " AND "), query.Limit)
	return SearchPlan{Statement: Statement{SQL: sql, Params: params}, Mode: mode, Type: searchType}, nil
}

func BuildFullTextPlan(config Config, query Query) (SearchPlan, error) {
	if err := config.Validate(); err != nil {
		return SearchPlan{}, err
	}
	if !config.EnableFullText {
		return SearchPlan{}, fmt.Errorf("full-text search is disabled")
	}
	if err := validateLimit(query.Limit, config.MaxSearchLimit); err != nil {
		return SearchPlan{}, err
	}
	if err := validateQueryText(query.Text); err != nil {
		return SearchPlan{}, fmt.Errorf("full-text query: %w", err)
	}
	filters, params, err := buildFilters(query.Filter, false)
	if err != nil {
		return SearchPlan{}, err
	}
	filters = append([]string{"SEARCH(ContentTokens, @query)"}, filters...)
	params["query"] = query.Text
	sql := fmt.Sprintf(`SELECT Id, Content, ChunkType, SessionId, ProjectPath, ProjectName,
  Timestamp, FilePath, Operation, MachineId,
  1.0 - SCORE(ContentTokens, @query) AS Distance
FROM ConversationChunks
WHERE %s
ORDER BY SCORE(ContentTokens, @query) DESC
LIMIT %d`, strings.Join(filters, " AND "), query.Limit)
	return SearchPlan{Statement: Statement{SQL: sql, Params: params}, Mode: SearchExact, Type: SearchTypeFullText}, nil
}

func BuildHybridPlan(config Config, query Query, vectorIndexAvailable bool) (SearchPlan, error) {
	if err := config.Validate(); err != nil {
		return SearchPlan{}, err
	}
	if !config.EnableFullText {
		return SearchPlan{}, fmt.Errorf("hybrid search requires full-text search")
	}
	if err := validateQueryText(query.Text); err != nil {
		return SearchPlan{}, fmt.Errorf("hybrid query text: %w", err)
	}
	if err := validateLimit(query.Limit, config.MaxSearchLimit); err != nil {
		return SearchPlan{}, err
	}
	if err := validateVector(query.Vector); err != nil {
		return SearchPlan{}, fmt.Errorf("query vector: %w", err)
	}
	mode, err := chooseVectorMode(config, query, vectorIndexAvailable)
	if err != nil {
		return SearchPlan{}, err
	}
	vectorFilters, params, err := buildFilters(query.Filter, true)
	if err != nil {
		return SearchPlan{}, err
	}
	textFilters, _, err := buildFilters(query.Filter, false)
	if err != nil {
		return SearchPlan{}, err
	}
	textFilters = append([]string{"SEARCH(ContentTokens, @query)"}, textFilters...)
	params["query"] = query.Text
	params["query_vector"] = append([]float32(nil), query.Vector...)
	params["rrf_k"] = float64(config.RRFK)
	table := TableName
	distance := "COSINE_DISTANCE(Vector, @query_vector)"
	if mode == SearchANN {
		table = TableName + "@{FORCE_INDEX=" + VectorIndexName + "}"
		distance = fmt.Sprintf(`APPROX_COSINE_DISTANCE(Vector, @query_vector, options => JSON '{"num_leaves_to_search": %d}')`, config.NumLeavesToSearch)
	}
	candidateLimit := max(query.Limit, config.HybridCandidateLimit)
	sql := fmt.Sprintf(`WITH VectorCandidates AS (
  SELECT rank, chunk_id AS Id FROM UNNEST(ARRAY(
    SELECT Id FROM %s WHERE %s ORDER BY %s ASC LIMIT %d
  )) AS chunk_id WITH OFFSET AS rank
), TextCandidates AS (
  SELECT rank, chunk_id AS Id FROM UNNEST(ARRAY(
    SELECT Id FROM ConversationChunks WHERE %s
    ORDER BY SCORE(ContentTokens, @query) DESC LIMIT %d
  )) AS chunk_id WITH OFFSET AS rank
), FusedCandidates AS (
  SELECT Id, SUM(1.0 / (@rrf_k + rank + 1)) AS Score
  FROM (SELECT Id, rank FROM VectorCandidates UNION ALL SELECT Id, rank FROM TextCandidates)
  GROUP BY Id
)
SELECT c.Id, c.Content, c.ChunkType, c.SessionId, c.ProjectPath, c.ProjectName,
  c.Timestamp, c.FilePath, c.Operation, c.MachineId,
  1.0 - LEAST(1.0, f.Score / (2.0 / (@rrf_k + 1))) AS Distance
FROM FusedCandidates f JOIN ConversationChunks c ON c.Id = f.Id
ORDER BY f.Score DESC LIMIT %d`, table, strings.Join(vectorFilters, " AND "), distance,
		candidateLimit, strings.Join(textFilters, " AND "), candidateLimit, query.Limit)
	return SearchPlan{Statement: Statement{SQL: sql, Params: params}, Mode: mode, Type: SearchTypeHybrid}, nil
}

func BuildBackfillReadPlan(config Config, prefix string) (Statement, error) {
	if err := config.Validate(); err != nil {
		return Statement{}, err
	}
	if len(prefix) != 2 || strings.ToLower(prefix) != prefix || strings.IndexFunc(prefix, func(r rune) bool {
		return !strings.ContainsRune("0123456789abcdef", r)
	}) >= 0 {
		return Statement{}, fmt.Errorf("backfill prefix must be exactly two lowercase hexadecimal characters")
	}
	sql := fmt.Sprintf(`SELECT Id, Content, ChunkType, SessionId, ProjectPath, ProjectName,
  Timestamp, UserUuid, AssistantUuid, FilePath, Operation, Model, SourceFile,
  SourceLine, ParentChunkId, ChildChunkIds, MachineId
FROM ConversationChunks
WHERE Vector IS NULL AND STARTS_WITH(Id, @prefix)
LIMIT %d`, config.BackfillBatch)
	return Statement{SQL: sql, Params: map[string]any{"prefix": prefix}}, nil
}

func chooseVectorMode(config Config, query Query, vectorIndexAvailable bool) (SearchMode, error) {
	if query.Mode != SearchAuto && query.Mode != SearchExact && query.Mode != SearchANN {
		return "", fmt.Errorf("search mode must be explicitly auto, exact, or ann")
	}
	annReady := config.EnableANN && config.UseANN && vectorIndexAvailable
	if hasUnstoredFilters(query.Filter) {
		if query.Mode == SearchANN {
			return "", fmt.Errorf("ANN is not lawful with filters absent from stored index columns")
		}
		return SearchExact, nil
	}
	if query.Mode == SearchANN {
		if !annReady {
			return "", fmt.Errorf("ANN was requested but the vector index is unavailable")
		}
		return SearchANN, nil
	}
	if query.Mode == SearchAuto && annReady {
		return SearchANN, nil
	}
	return SearchExact, nil
}

func hasUnstoredFilters(filter Filter) bool {
	return filter.ProjectPath != "" || filter.FilePath != "" || filter.Operation != "" ||
		!filter.DateFrom.IsZero() || !filter.DateTo.IsZero()
}

func buildFilters(filter Filter, vector bool) ([]string, map[string]any, error) {
	if !filter.DateFrom.IsZero() && !filter.DateTo.IsZero() && filter.DateFrom.After(filter.DateTo) {
		return nil, nil, fmt.Errorf("date_from must not be after date_to")
	}
	filters := make([]string, 0, 10)
	if vector {
		filters = append(filters, "Vector IS NOT NULL")
	}
	params := map[string]any{}
	add := func(column, parameter, value string, limit int) error {
		if value == "" {
			return nil
		}
		if len(value) > limit {
			return fmt.Errorf("%s filter exceeds %d bytes", parameter, limit)
		}
		filters = append(filters, column+" = @"+parameter)
		params[parameter] = value
		return nil
	}
	for _, item := range []struct {
		column, parameter, value string
		limit                    int
	}{
		{"ProjectPath", "project_path", filter.ProjectPath, 8192},
		{"ChunkType", "chunk_type", filter.ChunkType, 32},
		{"SessionId", "session_id", filter.SessionID, 128},
		{"ProjectName", "project_name", filter.ProjectName, 256},
		{"MachineId", "machine_id", filter.MachineID, 256},
		{"Operation", "operation", filter.Operation, 32},
	} {
		if err := add(item.column, item.parameter, item.value, item.limit); err != nil {
			return nil, nil, err
		}
	}
	if filter.FilePath != "" {
		if len(filter.FilePath) > 8192 {
			return nil, nil, fmt.Errorf("file_path filter exceeds 8192 bytes")
		}
		filters = append(filters, "FilePath LIKE @file_path")
		params["file_path"] = "%" + filter.FilePath + "%"
	}
	if !filter.DateFrom.IsZero() {
		filters = append(filters, "Timestamp >= @date_from")
		params["date_from"] = filter.DateFrom
	}
	if !filter.DateTo.IsZero() {
		filters = append(filters, "Timestamp <= @date_to")
		params["date_to"] = filter.DateTo
	}
	if len(filters) == 0 {
		filters = append(filters, "TRUE")
	}
	return filters, params, nil
}

func validateLimit(limit, maximum int) error {
	if limit < 1 || limit > maximum {
		return fmt.Errorf("limit must be in [1,%d]", maximum)
	}
	return nil
}

func validateQueryText(text string) error {
	if strings.TrimSpace(text) == "" || len(text) > maxQueryTextBytes {
		return fmt.Errorf("must be nonempty and at most %d bytes", maxQueryTextBytes)
	}
	return nil
}

func validateVector(vector []float32) error {
	if len(vector) != VectorDimension {
		return fmt.Errorf("vector dimension must be exactly %d, got %d", VectorDimension, len(vector))
	}
	nonZero := false
	for index, value := range vector {
		if !isFinite(value) {
			return fmt.Errorf("vector value %d is not finite", index)
		}
		if math.Abs(float64(value)) >= 1e-10 {
			nonZero = true
		}
	}
	if nonZero == false {
		return fmt.Errorf("vector must not be zero")
	}
	return nil
}

func isFinite(value float32) bool {
	converted := float64(value)
	return !math.IsNaN(converted) && !math.IsInf(converted, 0)
}

func validateChunk(chunk Chunk) error {
	for _, field := range []struct {
		name    string
		value   string
		maximum int
	}{
		{"id", chunk.ID, 64}, {"content", chunk.Content, 8 << 20},
		{"chunk_type", chunk.ChunkType, 32}, {"session_id", chunk.SessionID, 128},
		{"project_path", chunk.ProjectPath, 8192}, {"project_name", chunk.ProjectName, 256},
		{"source_file", chunk.SourceFile, 8192},
	} {
		if field.value == "" || len(field.value) > field.maximum {
			return fmt.Errorf("%s must be nonempty and at most %d bytes", field.name, field.maximum)
		}
	}
	for _, field := range []struct {
		name    string
		value   string
		maximum int
	}{
		{"user_uuid", chunk.UserUUID, 128}, {"assistant_uuid", chunk.AssistantUUID, 128},
		{"file_path", chunk.FilePath, 8192}, {"operation", chunk.Operation, 32},
		{"model", chunk.Model, 256}, {"parent_chunk_id", chunk.ParentChunkID, 64},
		{"machine_id", chunk.MachineID, 256},
	} {
		if len(field.value) > field.maximum {
			return fmt.Errorf("%s exceeds %d bytes", field.name, field.maximum)
		}
	}
	if chunk.Timestamp.IsZero() {
		return fmt.Errorf("timestamp is required")
	}
	if chunk.SourceLine < 0 {
		return fmt.Errorf("source line must not be negative")
	}
	if len(chunk.ChildChunkIDs) > 10_000 {
		return fmt.Errorf("child chunk ids exceed 10000 entries")
	}
	for _, id := range chunk.ChildChunkIDs {
		if id == "" || len(id) > 64 {
			return fmt.Errorf("child chunk id must be nonempty and at most 64 bytes")
		}
	}
	return nil
}

func hasLowerHexShardPrefix(id string) bool {
	if len(id) < 2 {
		return false
	}
	for _, value := range id[:2] {
		if !strings.ContainsRune("0123456789abcdef", value) {
			return false
		}
	}
	return true
}

func chunkValues(chunk Chunk, includeVector bool) []any {
	values := []any{chunk.ID, chunk.Content}
	if includeVector {
		values = append(values, append([]float32(nil), chunk.Vector...))
	}
	return append(values, chunk.ChunkType, chunk.SessionID, chunk.ProjectPath, chunk.ProjectName,
		chunk.Timestamp.UTC(), nullable(chunk.UserUUID), nullable(chunk.AssistantUUID),
		nullable(chunk.FilePath), nullable(chunk.Operation), nullable(chunk.Model), chunk.SourceFile,
		chunk.SourceLine, nullable(chunk.ParentChunkID), append([]string(nil), chunk.ChildChunkIDs...),
		nullable(chunk.MachineID))
}

func chunkRow(chunk Chunk) RemoteEmbeddingRow {
	return RemoteEmbeddingRow{
		ID: chunk.ID, Content: chunk.Content, ChunkType: chunk.ChunkType,
		SessionID: chunk.SessionID, ProjectPath: chunk.ProjectPath,
		ProjectName: chunk.ProjectName, Timestamp: chunk.Timestamp.UTC(),
		UserUUID: optionalString(chunk.UserUUID), AssistantUUID: optionalString(chunk.AssistantUUID),
		FilePath: optionalString(chunk.FilePath), Operation: optionalString(chunk.Operation),
		Model: optionalString(chunk.Model), SourceFile: chunk.SourceFile,
		SourceLine: chunk.SourceLine, ParentChunkID: optionalString(chunk.ParentChunkID),
		ChildChunkIDs: append([]string(nil), chunk.ChildChunkIDs...),
		MachineID:     optionalString(chunk.MachineID),
	}
}

func nullable(value string) any {
	if value == "" {
		return nil
	}
	return value
}

func optionalString(value string) *string {
	if value == "" {
		return nil
	}
	copy := value
	return &copy
}

func parseResultRows(rows []Row, searchType SearchType) ([]Result, error) {
	results := make([]Result, 0, len(rows))
	for index, row := range rows {
		if len(row) != 11 {
			return nil, fmt.Errorf("result row %d has %d columns, want 11", index, len(row))
		}
		result := Result{SearchType: searchType}
		var err error
		if result.ID, err = requiredRowString(row[0], "id"); err != nil {
			return nil, fmt.Errorf("result row %d: %w", index, err)
		}
		if result.Content, err = requiredRowString(row[1], "content"); err != nil {
			return nil, fmt.Errorf("result row %d: %w", index, err)
		}
		if result.ChunkType, err = requiredRowString(row[2], "chunk_type"); err != nil {
			return nil, fmt.Errorf("result row %d: %w", index, err)
		}
		if result.SessionID, err = requiredRowString(row[3], "session_id"); err != nil {
			return nil, fmt.Errorf("result row %d: %w", index, err)
		}
		if result.ProjectPath, err = requiredRowString(row[4], "project_path"); err != nil {
			return nil, fmt.Errorf("result row %d: %w", index, err)
		}
		if result.ProjectName, err = requiredRowString(row[5], "project_name"); err != nil {
			return nil, fmt.Errorf("result row %d: %w", index, err)
		}
		var ok bool
		if result.Timestamp, ok = row[6].(time.Time); !ok || result.Timestamp.IsZero() {
			return nil, fmt.Errorf("result row %d: timestamp is invalid", index)
		}
		if result.FilePath, err = optionalRowString(row[7], "file_path"); err != nil {
			return nil, fmt.Errorf("result row %d: %w", index, err)
		}
		if result.Operation, err = optionalRowString(row[8], "operation"); err != nil {
			return nil, fmt.Errorf("result row %d: %w", index, err)
		}
		if result.MachineID, err = optionalRowString(row[9], "machine_id"); err != nil {
			return nil, fmt.Errorf("result row %d: %w", index, err)
		}
		switch value := row[10].(type) {
		case float64:
			result.Distance = value
		case float32:
			result.Distance = float64(value)
		default:
			return nil, fmt.Errorf("result row %d distance is not numeric", index)
		}
		if math.IsNaN(result.Distance) || math.IsInf(result.Distance, 0) {
			return nil, fmt.Errorf("result row %d distance is not finite", index)
		}
		results = append(results, result)
	}
	return results, nil
}

func parseEmbeddingRows(rows []Row) ([]float32, error) {
	if len(rows) != 1 || len(rows[0]) != 1 {
		return nil, fmt.Errorf("query embedding returned an invalid row shape")
	}
	var vector []float32
	switch values := rows[0][0].(type) {
	case []float32:
		vector = append([]float32(nil), values...)
	case []float64:
		vector = make([]float32, len(values))
		for index, value := range values {
			if math.IsNaN(value) || math.IsInf(value, 0) || value > math.MaxFloat32 || value < -math.MaxFloat32 {
				return nil, fmt.Errorf("query embedding value %d is invalid", index)
			}
			vector[index] = float32(value)
		}
	default:
		return nil, fmt.Errorf("query embedding vector has an invalid type")
	}
	if err := validateVector(vector); err != nil {
		return nil, fmt.Errorf("query embedding: %w", err)
	}
	return vector, nil
}

func requiredRowString(value any, name string) (string, error) {
	converted, ok := value.(string)
	if !ok || converted == "" {
		return "", fmt.Errorf("%s is not a nonempty string", name)
	}
	return converted, nil
}

func optionalRowString(value any, name string) (string, error) {
	if value == nil {
		return "", nil
	}
	converted, ok := value.(string)
	if !ok {
		return "", fmt.Errorf("%s is neither string nor NULL", name)
	}
	return converted, nil
}

func hexByte(value byte) string {
	return fmt.Sprintf("%02x", value)
}
