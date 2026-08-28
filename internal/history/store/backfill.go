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
		var ok bool
		if chunk.ID, ok = row[0].(string); !ok {
			return nil, fmt.Errorf("backfill row %d id is not string", index)
		}
		if chunk.Content, ok = row[1].(string); !ok {
			return nil, fmt.Errorf("backfill row %d content is not string", index)
		}
		chunk.ChunkType, _ = row[2].(string)
		chunk.SessionID, _ = row[3].(string)
		chunk.ProjectPath, _ = row[4].(string)
		chunk.ProjectName, _ = row[5].(string)
		if chunk.Timestamp, ok = row[6].(time.Time); !ok {
			return nil, fmt.Errorf("backfill row %d timestamp is invalid", index)
		}
		chunk.UserUUID, _ = row[7].(string)
		chunk.AssistantUUID, _ = row[8].(string)
		chunk.FilePath, _ = row[9].(string)
		chunk.Operation, _ = row[10].(string)
		chunk.Model, _ = row[11].(string)
		chunk.SourceFile, _ = row[12].(string)
		if chunk.SourceLine, ok = toInt64(row[13]); !ok {
			return nil, fmt.Errorf("backfill row %d source line is invalid", index)
		}
		chunk.ParentChunkID, _ = row[14].(string)
		chunk.ChildChunkIDs, _ = row[15].([]string)
		chunk.MachineID, _ = row[16].(string)
		chunks = append(chunks, chunk)
	}
	return chunks, nil
}
