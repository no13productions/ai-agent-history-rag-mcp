package ingest

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"math"
	"strings"
	"testing"
	"time"
)

const digestA = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

func baseRequest() Request {
	return Request{
		MachineID:  "machine-A",
		ClientName: "client-X",
		Source: SourceKey{
			Format:       SourceClaudeCodeJSONL,
			RootVolume:   1,
			RootObject:   2,
			RelativePath: "project/session.jsonl",
		},
		Cursor: Cursor{
			Kind:           CursorPhysicalLines,
			StartExclusive: 0,
			EndInclusive:   7,
			PrefixSHA256:   digestA,
		},
		Generation: 3,
		Chunks: []Chunk{{
			ID:          strings.Repeat("b", 64),
			Content:     "café <ok>",
			ChunkType:   "turn",
			SessionID:   "session-1",
			ProjectPath: "/work/project",
			ProjectName: "project",
			Timestamp: time.Date(
				2026, time.August, 28, 8, 34, 56, 123456789,
				time.FixedZone("EDT", -4*60*60),
			),
			SourceFile: "/history/project/session.jsonl",
			SourceLine: 2,
		}},
	}
}

func TestEncodeRequestCanonicalBytesAndIdentity(t *testing.T) {
	request := baseRequest()
	encoded, err := EncodeRequest(request)
	if err != nil {
		t.Fatalf("EncodeRequest: %v", err)
	}
	expectedBody := `{"schema_version":1,"machine_id":"machine-A","client_name":"client-X","source":{"format":"claude_code_jsonl","root_volume":"0000000000000001","root_object":"0000000000000002","relative_path":"project/session.jsonl"},"cursor":{"kind":"physical_lines","start_exclusive":0,"end_inclusive":7,"prefix_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},"generation":3,"chunks":[{"id":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","content":"café <ok>","chunk_type":"turn","session_id":"session-1","project_path":"/work/project","project_name":"project","timestamp":"2026-08-28T12:34:56.123456789Z","source_file":"/history/project/session.jsonl","source_line":2}]}`
	if got := string(encoded.Body()); got != expectedBody {
		t.Fatalf("canonical body mismatch\n got: %s\nwant: %s", got, expectedBody)
	}
	expectedPayload := `[{"id":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","content":"café <ok>","chunk_type":"turn","session_id":"session-1","project_path":"/work/project","project_name":"project","timestamp":"2026-08-28T12:34:56.123456789Z","source_file":"/history/project/session.jsonl","source_line":2}]`
	if got := string(encoded.Payload()); got != expectedPayload {
		t.Fatalf("canonical payload mismatch\n got: %s\nwant: %s", got, expectedPayload)
	}
	if encoded.ByteCount() != len(expectedBody) || encoded.PayloadByteCount() != len(expectedPayload) {
		t.Fatal("encoded byte counts do not match returned canonical bytes")
	}
	bodyDigest := sha256.Sum256([]byte(expectedBody))
	payloadDigest := sha256.Sum256([]byte(expectedPayload))
	if encoded.RequestSHA256() != bodyDigest || encoded.PayloadSHA256() != payloadDigest {
		t.Fatal("encoded digests do not bind exact canonical bytes")
	}
	if len(encoded.RecordID()) != sha256.Size*2 {
		t.Fatalf("record id length = %d", len(encoded.RecordID()))
	}
	if encoded.RecordID() != "8be82022a925f128fb2b95c227d9d4891faef37c7205cf0cebc86726c2610694" {
		t.Fatalf("record id = %s", encoded.RecordID())
	}
	if _, err := hex.DecodeString(encoded.RecordID()); err != nil {
		t.Fatalf("record id is not lowercase hex: %v", err)
	}
	repeated, err := EncodeRequest(request)
	if err != nil || repeated.RecordID() != encoded.RecordID() {
		t.Fatal("same semantic request did not produce one record identity")
	}

	body := encoded.Body()
	body[0] = '!'
	payload := encoded.Payload()
	payload[0] = '!'
	if string(encoded.Body()) != expectedBody || string(encoded.Payload()) != expectedPayload {
		t.Fatal("caller mutated encoded request through returned slices")
	}
}

func TestRecordIdentityChangesWithEveryDurableBinding(t *testing.T) {
	baseline := baseRequest()
	first, err := EncodeRequest(baseline)
	if err != nil {
		t.Fatal(err)
	}
	cases := map[string]func(*Request){
		"machine":    func(r *Request) { r.MachineID = "machine-B" },
		"client":     func(r *Request) { r.ClientName = "client-Y" },
		"format":     func(r *Request) { r.Source.Format = SourceCodexJSONL },
		"root":       func(r *Request) { r.Source.RootObject++ },
		"relative":   func(r *Request) { r.Source.RelativePath = "project/other.jsonl" },
		"item":       func(r *Request) { r.Source.ItemID = "conversation-1" },
		"cursor":     func(r *Request) { r.Cursor.EndInclusive++ },
		"prefix":     func(r *Request) { r.Cursor.PrefixSHA256 = strings.Repeat("c", 64) },
		"generation": func(r *Request) { r.Generation++ },
		"chunk":      func(r *Request) { r.Chunks[0].Content = "changed" },
	}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			candidate := baseRequest()
			mutate(&candidate)
			encoded, err := EncodeRequest(candidate)
			if err != nil {
				t.Fatalf("EncodeRequest: %v", err)
			}
			if encoded.RecordID() == first.RecordID() {
				t.Fatal("changed durable binding retained prior record identity")
			}
		})
	}
}

func TestCursorUnionRefusesAmbiguousAndInvalidValues(t *testing.T) {
	cases := map[string]Cursor{
		"unknown kind": {Kind: "lineish"},
		"physical backwards": {
			Kind: CursorPhysicalLines, StartExclusive: 8, EndInclusive: 7, PrefixSHA256: digestA,
		},
		"physical missing digest": {
			Kind: CursorPhysicalLines, EndInclusive: 7,
		},
		"physical with revision": {
			Kind: CursorPhysicalLines, EndInclusive: 7, PrefixSHA256: digestA, RevisionSHA256: digestA,
		},
		"physical beyond signed store range": {
			Kind: CursorPhysicalLines, EndInclusive: uint64(math.MaxInt64) + 1, PrefixSHA256: digestA,
		},
		"snapshot missing current": {Kind: CursorSnapshotRevision},
		"snapshot with line": {
			Kind: CursorSnapshotRevision, EndInclusive: 1, RevisionSHA256: digestA,
		},
		"snapshot unchanged": {
			Kind: CursorSnapshotRevision, PreviousRevisionSHA256: digestA, RevisionSHA256: digestA,
		},
		"uppercase digest": {
			Kind: CursorSnapshotRevision, RevisionSHA256: strings.Repeat("A", 64),
		},
	}
	for name, cursor := range cases {
		t.Run(name, func(t *testing.T) {
			request := baseRequest()
			request.Cursor = cursor
			if _, err := EncodeRequest(request); !errors.Is(err, ErrInvalidCursor) {
				t.Fatalf("error = %v, want ErrInvalidCursor", err)
			}
		})
	}

	request := baseRequest()
	request.Cursor = Cursor{
		Kind:                   CursorSnapshotRevision,
		PreviousRevisionSHA256: strings.Repeat("c", 64),
		RevisionSHA256:         digestA,
	}
	encoded, err := EncodeRequest(request)
	if err != nil {
		t.Fatal(err)
	}
	body := string(encoded.Body())
	if strings.Contains(body, "start_exclusive") || strings.Contains(body, "end_inclusive") || strings.Contains(body, "prefix_sha256") {
		t.Fatal("snapshot cursor was represented with physical-line fields")
	}
	if !strings.Contains(body, `"kind":"snapshot_revision"`) {
		t.Fatal("snapshot cursor kind missing")
	}

	request = baseRequest()
	request.Cursor.StartExclusive = 7
	request.Cursor.EndInclusive = 7
	if _, err := EncodeRequest(request); err != nil {
		t.Fatalf("empty readable physical interval refused: %v", err)
	}
}

func TestSourceAndChunkValidationFailClosed(t *testing.T) {
	tests := map[string]func(*Request){
		"unknown format": func(r *Request) { r.Source.Format = "claudish" },
		"empty root identity": func(r *Request) {
			r.Source.RootVolume = 0
			r.Source.RootObject = 0
		},
		"absolute relative path":  func(r *Request) { r.Source.RelativePath = "/escape" },
		"unclean relative path":   func(r *Request) { r.Source.RelativePath = "a/../b" },
		"backslash relative path": func(r *Request) { r.Source.RelativePath = `a\b` },
		"blank item id":           func(r *Request) { r.Source.ItemID = "   " },
		"invalid machine":         func(r *Request) { r.MachineID = "bad machine" },
		"client control":          func(r *Request) { r.ClientName = "bad\nclient" },
		"empty id":                func(r *Request) { r.Chunks[0].ID = "" },
		"uppercase id":            func(r *Request) { r.Chunks[0].ID = strings.Repeat("B", 64) },
		"empty content":           func(r *Request) { r.Chunks[0].Content = "" },
		"content C1 control":      func(r *Request) { r.Chunks[0].Content = "bad\u0085content" },
		"chunk type control":      func(r *Request) { r.Chunks[0].ChunkType = "turn\nforged" },
		"optional metadata control": func(r *Request) {
			r.Chunks[0].Model = "bad\u0085model"
		},
		"source file control": func(r *Request) { r.Chunks[0].SourceFile = "/history/a\nforged" },
		"source line beyond signed store range": func(r *Request) {
			r.Chunks[0].SourceLine = uint64(math.MaxInt64) + 1
		},
		"zero timestamp": func(r *Request) { r.Chunks[0].Timestamp = time.Time{} },
		"timestamp before RFC3339 range": func(r *Request) {
			r.Chunks[0].Timestamp = time.Date(0, time.January, 1, 0, 0, 0, 0, time.UTC)
		},
		"timestamp after RFC3339 range": func(r *Request) {
			r.Chunks[0].Timestamp = time.Date(10000, time.January, 1, 0, 0, 0, 0, time.UTC)
		},
		"self parent": func(r *Request) { r.Chunks[0].ParentChunkID = r.Chunks[0].ID },
		"self child":  func(r *Request) { r.Chunks[0].ChildChunkIDs = []string{r.Chunks[0].ID} },
		"duplicate child": func(r *Request) {
			r.Chunks[0].ChildChunkIDs = []string{strings.Repeat("c", 64), strings.Repeat("c", 64)}
		},
		"duplicate chunk": func(r *Request) { r.Chunks = append(r.Chunks, r.Chunks[0]) },
	}
	for name, mutate := range tests {
		t.Run(name, func(t *testing.T) {
			request := baseRequest()
			mutate(&request)
			if _, err := EncodeRequest(request); err == nil {
				t.Fatal("invalid request was accepted")
			}
		})
	}

	request := baseRequest()
	second := request.Chunks[0]
	second.ID = strings.Repeat("c", 64)
	second.SourceFile = "/history/project/other.jsonl"
	request.Chunks = append(request.Chunks, second)
	if _, err := EncodeRequest(request); !errors.Is(err, ErrInvalidChunk) {
		t.Fatalf("mixed source-file provenance error = %v, want ErrInvalidChunk", err)
	}

	request = baseRequest()
	request.Chunks[0].Content = "first line\nsecond line\tvalue"
	if _, err := EncodeRequest(request); err != nil {
		t.Fatalf("multiline content refused: %v", err)
	}
}

func TestRequestChunkAndByteBounds(t *testing.T) {
	if MaxChunksPerRequest != 500 {
		t.Fatalf("chunk ceiling = %d", MaxChunksPerRequest)
	}
	if MaxRequestBytes != 1_048_576 {
		t.Fatalf("request byte ceiling = %d", MaxRequestBytes)
	}
	request := baseRequest()
	request.Chunks = nil
	if _, err := EncodeRequest(request); err != nil {
		t.Fatalf("readable zero-chunk interval refused: %v", err)
	}

	request = baseRequest()
	request.Chunks = make([]Chunk, MaxChunksPerRequest)
	for index := range request.Chunks {
		chunk := baseRequest().Chunks[0]
		chunk.ID = fmt.Sprintf("%064x", index+1)
		request.Chunks[index] = chunk
	}
	if _, err := EncodeRequest(request); err != nil {
		t.Fatalf("exact chunk ceiling refused: %v", err)
	}
	request.Chunks = append(request.Chunks, baseRequest().Chunks[0])
	request.Chunks[len(request.Chunks)-1].ID = strings.Repeat("f", 64)
	if _, err := EncodeRequest(request); !errors.Is(err, ErrChunkLimit) {
		t.Fatalf("over chunk ceiling error = %v", err)
	}

	request = baseRequest()
	first, err := EncodeRequest(request)
	if err != nil {
		t.Fatal(err)
	}
	delta := MaxRequestBytes - len(first.Body())
	request.Chunks[0].Content += strings.Repeat("x", delta)
	exact, err := EncodeRequest(request)
	if err != nil {
		t.Fatalf("exact byte ceiling refused: %v", err)
	}
	if len(exact.Body()) != MaxRequestBytes {
		t.Fatalf("body bytes = %d, want %d", len(exact.Body()), MaxRequestBytes)
	}
	request.Chunks[0].Content += "x"
	if _, err := EncodeRequest(request); !errors.Is(err, ErrRequestLimit) {
		t.Fatalf("over byte ceiling error = %v", err)
	}
}

func TestDeriveChunkIDBindsSourceAndSemanticCoordinates(t *testing.T) {
	identity := ChunkIdentity{
		MachineID:     "machine-A",
		Source:        baseRequest().Source,
		ChunkType:     "turn",
		SessionID:     "session-1",
		SourceLine:    2,
		Ordinal:       0,
		ContentSHA256: sha256.Sum256([]byte("café <ok>")),
	}
	first, err := DeriveChunkID(identity)
	if err != nil {
		t.Fatal(err)
	}
	second, err := DeriveChunkID(identity)
	if err != nil || first != second {
		t.Fatal("chunk identity is not deterministic")
	}
	identity.Ordinal++
	changed, err := DeriveChunkID(identity)
	if err != nil {
		t.Fatal(err)
	}
	if first == changed {
		t.Fatal("semantic ordinal was not bound into chunk identity")
	}
	identity.Ordinal = 0
	identity.MachineID = "machine-B"
	changed, err = DeriveChunkID(identity)
	if err != nil {
		t.Fatal(err)
	}
	if first == changed {
		t.Fatal("machine identity was not bound into chunk identity")
	}
	identity.MachineID = "machine-A"
	identity.ContentSHA256 = [sha256.Size]byte{}
	if _, err := DeriveChunkID(identity); !errors.Is(err, ErrInvalidChunk) {
		t.Fatalf("zero content digest error = %v, want ErrInvalidChunk", err)
	}
	identity.ContentSHA256 = sha256.Sum256([]byte("café <ok>"))
	identity.MachineID = "bad machine"
	if _, err := DeriveChunkID(identity); !errors.Is(err, ErrInvalidChunk) {
		t.Fatalf("invalid machine error = %v, want ErrInvalidChunk", err)
	}
	identity.MachineID = "machine-A"
	identity.SourceLine = uint64(math.MaxInt64) + 1
	if _, err := DeriveChunkID(identity); !errors.Is(err, ErrInvalidChunk) {
		t.Fatalf("source-line overflow error = %v, want ErrInvalidChunk", err)
	}
	if first != "e2362143798dddf56f20d1b36688b57e7117f28a3d4a00a585a9407ec0e6d435" {
		t.Fatalf("chunk id = %s", first)
	}
}
