package ingest

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"path"
	"strings"
	"time"
	"unicode"
	"unicode/utf8"
)

const (
	MaxChunksPerRequest  = 500
	MaxRequestBytes      = 1 << 20
	recordIdentityDomain = "history-rag/outbox-record/v1\x00"
	chunkIdentityDomain  = "history-rag/chunk/v1\x00"
)

var (
	ErrInvalidRequest = errors.New("invalid ingestion request")
	ErrInvalidSource  = errors.New("invalid ingestion source")
	ErrInvalidCursor  = errors.New("invalid ingestion cursor")
	ErrInvalidChunk   = errors.New("invalid ingestion chunk")
	ErrChunkLimit     = errors.New("ingestion chunk limit exceeded")
	ErrRequestLimit   = errors.New("ingestion request byte limit exceeded")
)

type SourceFormat string

const (
	SourceClaudeCodeJSONL       SourceFormat = "claude_code_jsonl"
	SourceCodexJSONL            SourceFormat = "codex_jsonl"
	SourceGeminiCLIJSON         SourceFormat = "gemini_cli_json"
	SourceAntigravityTranscript SourceFormat = "antigravity_transcript_or_protobuf"
	SourceChatGPTExportJSON     SourceFormat = "chatgpt_export_json"
	SourceClaudeAppExportJSON   SourceFormat = "claude_app_export_json"
)

type SourceKey struct {
	Format       SourceFormat
	RootVolume   uint64
	RootObject   uint64
	RelativePath string
	ItemID       string
}

type CursorKind string

const (
	CursorPhysicalLines    CursorKind = "physical_lines"
	CursorSnapshotRevision CursorKind = "snapshot_revision"
)

type Cursor struct {
	Kind                   CursorKind
	StartExclusive         uint64
	EndInclusive           uint64
	PrefixSHA256           string
	PreviousRevisionSHA256 string
	RevisionSHA256         string
}

type Chunk struct {
	ID            string
	Content       string
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
	SourceLine    uint64
	ParentChunkID string
	ChildChunkIDs []string
}

type Request struct {
	MachineID  string
	ClientName string
	Source     SourceKey
	Cursor     Cursor
	Generation uint64
	Chunks     []Chunk
}

type ChunkIdentity struct {
	MachineID     string
	Source        SourceKey
	ChunkType     string
	SessionID     string
	SourceLine    uint64
	Ordinal       uint32
	ContentSHA256 [sha256.Size]byte
}

type EncodedRequest struct {
	body          []byte
	payload       []byte
	requestSHA256 [sha256.Size]byte
	payloadSHA256 [sha256.Size]byte
	recordID      string
	chunkCount    int
}

func (r EncodedRequest) Body() []byte {
	return append([]byte(nil), r.body...)
}

func (r EncodedRequest) Payload() []byte {
	return append([]byte(nil), r.payload...)
}

func (r EncodedRequest) RequestSHA256() [sha256.Size]byte { return r.requestSHA256 }
func (r EncodedRequest) PayloadSHA256() [sha256.Size]byte { return r.payloadSHA256 }
func (r EncodedRequest) RecordID() string                 { return r.recordID }
func (r EncodedRequest) ChunkCount() int                  { return r.chunkCount }
func (r EncodedRequest) ByteCount() int                   { return len(r.body) }
func (r EncodedRequest) PayloadByteCount() int            { return len(r.payload) }

type sourceWire struct {
	Format       SourceFormat `json:"format"`
	RootVolume   string       `json:"root_volume"`
	RootObject   string       `json:"root_object"`
	RelativePath string       `json:"relative_path"`
	ItemID       string       `json:"item_id,omitempty"`
}

type cursorWire struct {
	Kind                   CursorKind `json:"kind"`
	StartExclusive         *uint64    `json:"start_exclusive,omitempty"`
	EndInclusive           *uint64    `json:"end_inclusive,omitempty"`
	PrefixSHA256           string     `json:"prefix_sha256,omitempty"`
	PreviousRevisionSHA256 string     `json:"previous_revision_sha256,omitempty"`
	RevisionSHA256         string     `json:"revision_sha256,omitempty"`
}

type chunkWire struct {
	ID            string   `json:"id"`
	Content       string   `json:"content"`
	ChunkType     string   `json:"chunk_type"`
	SessionID     string   `json:"session_id"`
	ProjectPath   string   `json:"project_path"`
	ProjectName   string   `json:"project_name"`
	Timestamp     string   `json:"timestamp"`
	UserUUID      string   `json:"user_uuid,omitempty"`
	AssistantUUID string   `json:"assistant_uuid,omitempty"`
	FilePath      string   `json:"file_path,omitempty"`
	Operation     string   `json:"operation,omitempty"`
	Model         string   `json:"model,omitempty"`
	SourceFile    string   `json:"source_file"`
	SourceLine    uint64   `json:"source_line"`
	ParentChunkID string   `json:"parent_chunk_id,omitempty"`
	ChildChunkIDs []string `json:"child_chunk_ids,omitempty"`
}

type requestWire struct {
	SchemaVersion int             `json:"schema_version"`
	MachineID     string          `json:"machine_id"`
	ClientName    string          `json:"client_name,omitempty"`
	Source        sourceWire      `json:"source"`
	Cursor        cursorWire      `json:"cursor"`
	Generation    uint64          `json:"generation"`
	Chunks        json.RawMessage `json:"chunks"`
}

type chunkIdentityWire struct {
	SchemaVersion int        `json:"schema_version"`
	MachineID     string     `json:"machine_id"`
	Source        sourceWire `json:"source"`
	ChunkType     string     `json:"chunk_type"`
	SessionID     string     `json:"session_id"`
	SourceLine    uint64     `json:"source_line"`
	Ordinal       uint32     `json:"ordinal"`
	ContentSHA256 string     `json:"content_sha256"`
}

func EncodeRequest(request Request) (EncodedRequest, error) {
	if len(request.Chunks) > MaxChunksPerRequest {
		return EncodedRequest{}, ErrChunkLimit
	}
	if !validMachineID(request.MachineID) {
		return EncodedRequest{}, fmt.Errorf("%w: machine_id", ErrInvalidRequest)
	}
	if request.ClientName != "" && !validText(request.ClientName, 256, false) {
		return EncodedRequest{}, fmt.Errorf("%w: client_name", ErrInvalidRequest)
	}
	source, err := request.Source.wire()
	if err != nil {
		return EncodedRequest{}, err
	}
	cursor, err := request.Cursor.wire()
	if err != nil {
		return EncodedRequest{}, err
	}

	chunks := make([]chunkWire, 0, len(request.Chunks))
	seen := make(map[string]struct{}, len(request.Chunks))
	var sourceFile string
	for index := range request.Chunks {
		chunk, err := request.Chunks[index].wire()
		if err != nil {
			return EncodedRequest{}, fmt.Errorf("%w: chunks[%d]: %v", ErrInvalidChunk, index, err)
		}
		if _, exists := seen[chunk.ID]; exists {
			return EncodedRequest{}, fmt.Errorf("%w: duplicate chunk id", ErrInvalidChunk)
		}
		if index == 0 {
			sourceFile = chunk.SourceFile
		} else if chunk.SourceFile != sourceFile {
			return EncodedRequest{}, fmt.Errorf("%w: mixed source_file provenance", ErrInvalidChunk)
		}
		seen[chunk.ID] = struct{}{}
		chunks = append(chunks, chunk)
	}
	payload, err := marshalCanonical(chunks)
	if err != nil {
		return EncodedRequest{}, fmt.Errorf("%w: encode chunks: %v", ErrInvalidRequest, err)
	}
	body, err := marshalCanonical(requestWire{
		SchemaVersion: 1,
		MachineID:     request.MachineID,
		ClientName:    request.ClientName,
		Source:        source,
		Cursor:        cursor,
		Generation:    request.Generation,
		Chunks:        json.RawMessage(payload),
	})
	if err != nil {
		return EncodedRequest{}, fmt.Errorf("%w: encode request: %v", ErrInvalidRequest, err)
	}
	if len(body) > MaxRequestBytes {
		return EncodedRequest{}, ErrRequestLimit
	}
	requestDigest := sha256.Sum256(body)
	payloadDigest := sha256.Sum256(payload)
	identityHasher := sha256.New()
	_, _ = identityHasher.Write([]byte(recordIdentityDomain))
	_, _ = identityHasher.Write(requestDigest[:])
	return EncodedRequest{
		body:          body,
		payload:       payload,
		requestSHA256: requestDigest,
		payloadSHA256: payloadDigest,
		recordID:      hex.EncodeToString(identityHasher.Sum(nil)),
		chunkCount:    len(chunks),
	}, nil
}

func DeriveChunkID(identity ChunkIdentity) (string, error) {
	if !validMachineID(identity.MachineID) {
		return "", fmt.Errorf("%w: machine_id", ErrInvalidChunk)
	}
	source, err := identity.Source.wire()
	if err != nil {
		return "", err
	}
	if !validText(identity.ChunkType, 128, false) {
		return "", fmt.Errorf("%w: chunk_type", ErrInvalidChunk)
	}
	if !validText(identity.SessionID, 512, false) {
		return "", fmt.Errorf("%w: session_id", ErrInvalidChunk)
	}
	if identity.SourceLine > math.MaxInt64 || identity.ContentSHA256 == ([sha256.Size]byte{}) {
		return "", ErrInvalidChunk
	}
	wire := chunkIdentityWire{
		SchemaVersion: 1,
		MachineID:     identity.MachineID,
		Source:        source,
		ChunkType:     identity.ChunkType,
		SessionID:     identity.SessionID,
		SourceLine:    identity.SourceLine,
		Ordinal:       identity.Ordinal,
		ContentSHA256: hex.EncodeToString(identity.ContentSHA256[:]),
	}
	encoded, err := marshalCanonical(wire)
	if err != nil {
		return "", fmt.Errorf("%w: encode chunk identity: %v", ErrInvalidChunk, err)
	}
	hasher := sha256.New()
	_, _ = hasher.Write([]byte(chunkIdentityDomain))
	_, _ = hasher.Write(encoded)
	return hex.EncodeToString(hasher.Sum(nil)), nil
}

func (s SourceKey) wire() (sourceWire, error) {
	if !validSourceFormat(s.Format) || (s.RootVolume == 0 && s.RootObject == 0) {
		return sourceWire{}, ErrInvalidSource
	}
	if !validRelativePath(s.RelativePath) {
		return sourceWire{}, ErrInvalidSource
	}
	if s.ItemID != "" && !validText(s.ItemID, 512, false) {
		return sourceWire{}, ErrInvalidSource
	}
	return sourceWire{
		Format:       s.Format,
		RootVolume:   fmt.Sprintf("%016x", s.RootVolume),
		RootObject:   fmt.Sprintf("%016x", s.RootObject),
		RelativePath: s.RelativePath,
		ItemID:       s.ItemID,
	}, nil
}

func (c Cursor) wire() (cursorWire, error) {
	switch c.Kind {
	case CursorPhysicalLines:
		if c.StartExclusive > math.MaxInt64 || c.EndInclusive > math.MaxInt64 ||
			c.EndInclusive < c.StartExclusive || !validSHA256(c.PrefixSHA256) ||
			c.PreviousRevisionSHA256 != "" || c.RevisionSHA256 != "" {
			return cursorWire{}, ErrInvalidCursor
		}
		start := c.StartExclusive
		end := c.EndInclusive
		return cursorWire{
			Kind:           c.Kind,
			StartExclusive: &start,
			EndInclusive:   &end,
			PrefixSHA256:   c.PrefixSHA256,
		}, nil
	case CursorSnapshotRevision:
		if c.StartExclusive != 0 || c.EndInclusive != 0 || c.PrefixSHA256 != "" ||
			!validSHA256(c.RevisionSHA256) ||
			(c.PreviousRevisionSHA256 != "" && !validSHA256(c.PreviousRevisionSHA256)) ||
			c.PreviousRevisionSHA256 == c.RevisionSHA256 {
			return cursorWire{}, ErrInvalidCursor
		}
		return cursorWire{
			Kind:                   c.Kind,
			PreviousRevisionSHA256: c.PreviousRevisionSHA256,
			RevisionSHA256:         c.RevisionSHA256,
		}, nil
	default:
		return cursorWire{}, ErrInvalidCursor
	}
}

func (c Chunk) wire() (chunkWire, error) {
	if !validSHA256(c.ID) || !validText(c.Content, MaxRequestBytes, true) ||
		!validText(c.ChunkType, 128, false) || !validText(c.SessionID, 512, false) ||
		!validText(c.ProjectPath, 4096, false) || !validText(c.ProjectName, 512, false) ||
		!validTimestamp(c.Timestamp) || !validText(c.SourceFile, 4096, false) || c.SourceLine > math.MaxInt64 {
		return chunkWire{}, ErrInvalidChunk
	}
	if !validOptionalText(c.UserUUID, 512) || !validOptionalText(c.AssistantUUID, 512) ||
		!validOptionalText(c.FilePath, 4096) || !validOptionalText(c.Operation, 128) ||
		!validOptionalText(c.Model, 512) {
		return chunkWire{}, ErrInvalidChunk
	}
	if c.ParentChunkID != "" && (!validSHA256(c.ParentChunkID) || c.ParentChunkID == c.ID) {
		return chunkWire{}, ErrInvalidChunk
	}
	children := append([]string(nil), c.ChildChunkIDs...)
	childSet := make(map[string]struct{}, len(children))
	for _, child := range children {
		if !validSHA256(child) || child == c.ID {
			return chunkWire{}, ErrInvalidChunk
		}
		if _, exists := childSet[child]; exists {
			return chunkWire{}, ErrInvalidChunk
		}
		childSet[child] = struct{}{}
	}
	return chunkWire{
		ID:            c.ID,
		Content:       c.Content,
		ChunkType:     c.ChunkType,
		SessionID:     c.SessionID,
		ProjectPath:   c.ProjectPath,
		ProjectName:   c.ProjectName,
		Timestamp:     c.Timestamp.UTC().Format(time.RFC3339Nano),
		UserUUID:      c.UserUUID,
		AssistantUUID: c.AssistantUUID,
		FilePath:      c.FilePath,
		Operation:     c.Operation,
		Model:         c.Model,
		SourceFile:    c.SourceFile,
		SourceLine:    c.SourceLine,
		ParentChunkID: c.ParentChunkID,
		ChildChunkIDs: children,
	}, nil
}

func marshalCanonical(value any) ([]byte, error) {
	var buffer bytes.Buffer
	encoder := json.NewEncoder(&buffer)
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(value); err != nil {
		return nil, err
	}
	encoded := buffer.Bytes()
	if len(encoded) == 0 || encoded[len(encoded)-1] != '\n' {
		return nil, errors.New("canonical JSON encoder omitted terminator")
	}
	return append([]byte(nil), encoded[:len(encoded)-1]...), nil
}

func validSourceFormat(format SourceFormat) bool {
	switch format {
	case SourceClaudeCodeJSONL, SourceCodexJSONL, SourceGeminiCLIJSON,
		SourceAntigravityTranscript, SourceChatGPTExportJSON, SourceClaudeAppExportJSON:
		return true
	default:
		return false
	}
}

func validRelativePath(value string) bool {
	return validText(value, 4096, false) && value != "." && !path.IsAbs(value) &&
		path.Clean(value) == value && !strings.Contains(value, "\\")
}

func validMachineID(value string) bool {
	if len(value) == 0 || len(value) > 128 {
		return false
	}
	for index := 0; index < len(value); index++ {
		character := value[index]
		if (character >= 'a' && character <= 'z') || (character >= 'A' && character <= 'Z') ||
			(character >= '0' && character <= '9') || strings.ContainsRune("_.:-", rune(character)) {
			continue
		}
		return false
	}
	return true
}

func validOptionalText(value string, maximum int) bool {
	return value == "" || validText(value, maximum, false)
}

func validTimestamp(value time.Time) bool {
	year := value.UTC().Year()
	return !value.IsZero() && year >= 1 && year <= 9999
}

func validText(value string, maximum int, allowLineBreaks bool) bool {
	if strings.TrimSpace(value) == "" || len(value) > maximum || !utf8.ValidString(value) {
		return false
	}
	for _, character := range value {
		if !unicode.IsControl(character) {
			continue
		}
		if allowLineBreaks && (character == '\n' || character == '\r' || character == '\t') {
			continue
		}
		return false
	}
	return true
}

func validSHA256(value string) bool {
	if len(value) != sha256.Size*2 {
		return false
	}
	for index := 0; index < len(value); index++ {
		character := value[index]
		if (character < '0' || character > '9') && (character < 'a' || character > 'f') {
			return false
		}
	}
	return true
}
