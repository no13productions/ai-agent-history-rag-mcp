package store

import (
	"context"
	"time"
)

const (
	TableName              = "ConversationChunks"
	ContentTokensColumn    = "ContentTokens"
	ContentSearchIndex     = "ConversationChunksContentSearch"
	VectorIndexName        = "ConversationChunksVectorIndex"
	RemoteModelName        = "ConversationEmbeddingModel"
	EmbeddingModelName     = "gemini-embedding-001"
	VectorDimension        = 3072
	VectorIndexThreshold   = 10_000
	TaskRetrievalQuery     = "RETRIEVAL_QUERY"
	TaskRetrievalDocument  = "RETRIEVAL_DOCUMENT"
	TaskSemanticSimilarity = "SEMANTIC_SIMILARITY"
)

type EmbeddingStrategy string

const (
	EmbeddingRemoteModel EmbeddingStrategy = "remote_model"
	EmbeddingDeferred    EmbeddingStrategy = "deferred"
)

type SearchMode string

const (
	SearchAuto  SearchMode = "auto"
	SearchExact SearchMode = "exact"
	SearchANN   SearchMode = "ann"
)

type SearchType string

const (
	SearchTypeExact    SearchType = "exact"
	SearchTypeANN      SearchType = "ann"
	SearchTypeFullText SearchType = "full_text"
	SearchTypeHybrid   SearchType = "hybrid"
)

type Chunk struct {
	ID            string
	Content       string
	Vector        []float32
	ChunkType     string
	SessionID     string
	ProjectPath   string
	ProjectName   string
	Timestamp     time.Time
	UserUUID      string
	AssistantUUID string
	FilePath      string
	Operation     string
	Model         string
	SourceFile    string
	SourceLine    int64
	ParentChunkID string
	ChildChunkIDs []string
	MachineID     string
}

type RemoteEmbeddingRow struct {
	ID            string    `spanner:"id"`
	Content       string    `spanner:"content"`
	ChunkType     string    `spanner:"chunk_type"`
	SessionID     string    `spanner:"session_id"`
	ProjectPath   string    `spanner:"project_path"`
	ProjectName   string    `spanner:"project_name"`
	Timestamp     time.Time `spanner:"timestamp"`
	UserUUID      *string   `spanner:"user_uuid"`
	AssistantUUID *string   `spanner:"assistant_uuid"`
	FilePath      *string   `spanner:"file_path"`
	Operation     *string   `spanner:"operation"`
	Model         *string   `spanner:"model"`
	SourceFile    string    `spanner:"source_file"`
	SourceLine    int64     `spanner:"source_line"`
	ParentChunkID *string   `spanner:"parent_chunk_id"`
	ChildChunkIDs []string  `spanner:"child_chunk_ids"`
	MachineID     *string   `spanner:"machine_id"`
}

type Filter struct {
	ProjectPath string
	ChunkType   string
	SessionID   string
	ProjectName string
	MachineID   string
	FilePath    string
	Operation   string
	DateFrom    time.Time
	DateTo      time.Time
}

type Query struct {
	Text   string
	Vector []float32
	Limit  int
	Mode   SearchMode
	Filter Filter
}

type Result struct {
	ID          string
	Content     string
	ChunkType   string
	SessionID   string
	ProjectPath string
	ProjectName string
	Timestamp   time.Time
	FilePath    string
	Operation   string
	MachineID   string
	Distance    float64
	SearchType  SearchType
}

type Stats struct {
	TotalChunks        int64
	EmbeddedChunks     int64
	AwaitingEmbedding  int64
	Backend            string
	Project            string
	Instance           string
	Database           string
	Dimension          int
	FullTextEnabled    bool
	VectorIndexEnabled bool
	VectorSearchMode   SearchMode
	EmbeddingStrategy  EmbeddingStrategy
	EmbeddingModel     string
}

type BackfillReport struct {
	Embedded int64
	Failures map[string]string
}

type Statement struct {
	SQL    string
	Params map[string]any
}

type Row []any

type Mutation struct {
	Table   string
	Columns []string
	Values  [][]any
}

type DDLStatement struct {
	SQL             string
	AlreadyExistsOK bool
}

type Executor interface {
	Query(context.Context, Statement) ([]Row, error)
	Execute(context.Context, Statement) (int64, error)
	Apply(context.Context, Mutation) error
	ReadWrite(context.Context, func(Transaction) error) error
	UpdateDDL(context.Context, []DDLStatement) error
	Close() error
}

type Transaction interface {
	Execute(context.Context, Statement) (int64, error)
	Apply(context.Context, Mutation) error
}

type Store interface {
	Initialize(context.Context) error
	Upsert(context.Context, []Chunk) error
	Search(context.Context, Query) ([]Result, error)
	HybridSearch(context.Context, Query) ([]Result, error)
	Stats(context.Context) (Stats, error)
	DeleteMachine(context.Context, string) (int64, error)
	Clear(context.Context) (int64, error)
	Optimize(context.Context) error
	Close() error
}

type UpsertPlan struct {
	Statement *Statement
	Mutation  *Mutation
}

type SearchPlan struct {
	Statement Statement
	Mode      SearchMode
	Type      SearchType
}
