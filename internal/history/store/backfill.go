package store

import (
	"context"
	"fmt"
	"sync"
	"time"
)

func BackfillShards() []string {
	shards := make([]string, 256)
	for value := range 256 {
		shards[value] = fmt.Sprintf("%02x", value)
	}
	return shards
}

func (s *SpannerStore) Backfill(ctx context.Context) (BackfillReport, error) {
	if err := contextError(ctx); err != nil {
		return BackfillReport{}, err
	}
	if s.config.EmbeddingStrategy != EmbeddingDeferred {
		return BackfillReport{}, fmt.Errorf("backfill requires deferred embedding strategy")
	}
	type shardResult struct {
		prefix string
		count  int64
		err    error
	}
	jobs := make(chan string)
	results := make(chan shardResult, len(BackfillShards()))
	var workers sync.WaitGroup
	for range s.config.BackfillConcurrency {
		workers.Add(1)
		go func() {
			defer workers.Done()
			for prefix := range jobs {
				if err := contextError(ctx); err != nil {
					results <- shardResult{prefix: prefix, err: err}
					continue
				}
				count, err := s.backfillShard(ctx, prefix)
				results <- shardResult{prefix: prefix, count: count, err: err}
			}
		}()
	}
	go func() {
		defer close(jobs)
		for _, prefix := range BackfillShards() {
			select {
			case jobs <- prefix:
			case <-ctx.Done():
				return
			}
		}
	}()
	go func() {
		workers.Wait()
		close(results)
	}()
	report := BackfillReport{Failures: map[string]string{}}
	completed := 0
	for result := range results {
		completed++
		report.Embedded += result.count
		if result.err != nil {
			report.Failures[result.prefix] = result.err.Error()
		}
	}
	if err := contextError(ctx); err != nil {
		return report, err
	}
	if completed != 256 {
		return report, fmt.Errorf("backfill completed %d of 256 shards", completed)
	}
	if report.Embedded > 0 {
		s.invalidateStats()
	}
	return report, nil
}

func (s *SpannerStore) runBackfillShard(ctx context.Context, prefix string) (int64, error) {
	var embedded int64
	for batch := 0; batch < s.config.BackfillMaxBatchesPerShard; batch++ {
		if err := contextError(ctx); err != nil {
			return embedded, err
		}
		statement, err := BuildBackfillReadPlan(s.config, prefix)
		if err != nil {
			return embedded, err
		}
		rows, err := s.executor.Query(ctx, statement)
		if err != nil {
			return embedded, fmt.Errorf("read NULL-vector batch: %w", err)
		}
		if len(rows) == 0 {
			return embedded, nil
		}
		chunks, err := backfillRows(rows)
		if err != nil {
			return embedded, err
		}
		remoteConfig := s.config
		remoteConfig.EmbeddingStrategy = EmbeddingRemoteModel
		plan, err := BuildUpsertPlan(remoteConfig, chunks)
		if err != nil {
			return embedded, err
		}
		if plan.Statement == nil {
			return embedded, fmt.Errorf("backfill remote-model plan is missing")
		}
		var count int64
		err = s.executor.ReadWrite(ctx, func(transaction Transaction) error {
			var transactionErr error
			count, transactionErr = transaction.Execute(ctx, *plan.Statement)
			return transactionErr
		})
		if err != nil {
			return embedded, fmt.Errorf("embed NULL-vector batch: %w", err)
		}
		if count < 0 || count > int64(len(chunks)) {
			return embedded, fmt.Errorf("remote-model affected count %d is invalid for batch %d", count, len(chunks))
		}
		embedded += count
	}
	return embedded, fmt.Errorf("backfill shard exceeded %d batches; remaining rows stay NULL", s.config.BackfillMaxBatchesPerShard)
}

func backfillRows(rows []Row) ([]Chunk, error) {
	chunks := make([]Chunk, 0, len(rows))
	for index, row := range rows {
		if len(row) != 17 {
			return nil, fmt.Errorf("backfill row %d has %d columns, want 17", index, len(row))
		}
		chunk := Chunk{}
		var err error
		if chunk.ID, err = requiredRowString(row[0], "id"); err != nil {
			return nil, fmt.Errorf("backfill row %d: %w", index, err)
		}
		if chunk.Content, err = requiredRowString(row[1], "content"); err != nil {
			return nil, fmt.Errorf("backfill row %d: %w", index, err)
		}
		if chunk.ChunkType, err = requiredRowString(row[2], "chunk_type"); err != nil {
			return nil, fmt.Errorf("backfill row %d: %w", index, err)
		}
		if chunk.SessionID, err = requiredRowString(row[3], "session_id"); err != nil {
			return nil, fmt.Errorf("backfill row %d: %w", index, err)
		}
		if chunk.ProjectPath, err = requiredRowString(row[4], "project_path"); err != nil {
			return nil, fmt.Errorf("backfill row %d: %w", index, err)
		}
		if chunk.ProjectName, err = requiredRowString(row[5], "project_name"); err != nil {
			return nil, fmt.Errorf("backfill row %d: %w", index, err)
		}
		var ok bool
		if chunk.Timestamp, ok = row[6].(time.Time); !ok {
			return nil, fmt.Errorf("backfill row %d timestamp is invalid", index)
		}
		if chunk.UserUUID, err = optionalRowString(row[7], "user_uuid"); err != nil {
			return nil, fmt.Errorf("backfill row %d: %w", index, err)
		}
		if chunk.AssistantUUID, err = optionalRowString(row[8], "assistant_uuid"); err != nil {
			return nil, fmt.Errorf("backfill row %d: %w", index, err)
		}
		if chunk.FilePath, err = optionalRowString(row[9], "file_path"); err != nil {
			return nil, fmt.Errorf("backfill row %d: %w", index, err)
		}
		if chunk.Operation, err = optionalRowString(row[10], "operation"); err != nil {
			return nil, fmt.Errorf("backfill row %d: %w", index, err)
		}
		if chunk.Model, err = optionalRowString(row[11], "model"); err != nil {
			return nil, fmt.Errorf("backfill row %d: %w", index, err)
		}
		if chunk.SourceFile, err = requiredRowString(row[12], "source_file"); err != nil {
			return nil, fmt.Errorf("backfill row %d: %w", index, err)
		}
		if chunk.SourceLine, ok = toInt64(row[13]); !ok {
			return nil, fmt.Errorf("backfill row %d source line is invalid", index)
		}
		if chunk.ParentChunkID, err = optionalRowString(row[14], "parent_chunk_id"); err != nil {
			return nil, fmt.Errorf("backfill row %d: %w", index, err)
		}
		switch childIDs := row[15].(type) {
		case nil:
			chunk.ChildChunkIDs = nil
		case []string:
			chunk.ChildChunkIDs = append([]string(nil), childIDs...)
		default:
			return nil, fmt.Errorf("backfill row %d child_chunk_ids is neither string array nor NULL", index)
		}
		if chunk.MachineID, err = optionalRowString(row[16], "machine_id"); err != nil {
			return nil, fmt.Errorf("backfill row %d: %w", index, err)
		}
		if err := validateChunk(chunk); err != nil {
			return nil, fmt.Errorf("backfill row %d: %w", index, err)
		}
		if !hasLowerHexShardPrefix(chunk.ID) {
			return nil, fmt.Errorf("backfill row %d id lacks a lowercase hexadecimal shard prefix", index)
		}
		chunks = append(chunks, chunk)
	}
	return chunks, nil
}
