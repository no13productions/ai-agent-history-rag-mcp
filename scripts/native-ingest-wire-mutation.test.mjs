#!/usr/bin/env node

import { execFileSync } from "node:child_process";
import { cpSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");

const mutations = [
  {
    name: "exact chunk ceiling",
    from: "MaxChunksPerRequest  = 500",
    to: "MaxChunksPerRequest  = 501",
  },
  {
    name: "exact request byte ceiling",
    from: "MaxRequestBytes      = 1 << 20",
    to: "MaxRequestBytes      = (1 << 20) + 1",
  },
  {
    name: "record identity domain",
    from: 'recordIdentityDomain = "history-rag/outbox-record/v1\\x00"',
    to: 'recordIdentityDomain = "history-rag/outbox-record/v2\\x00"',
  },
  {
    name: "chunk identity domain",
    from: 'chunkIdentityDomain  = "history-rag/chunk/v1\\x00"',
    to: 'chunkIdentityDomain  = "history-rag/chunk/v2\\x00"',
  },
  {
    name: "encoded body ownership",
    from: "return append([]byte(nil), r.body...)",
    to: "return r.body",
  },
  {
    name: "encoded payload ownership",
    from: "return append([]byte(nil), r.payload...)",
    to: "return r.payload",
  },
  {
    name: "encoded request byte count",
    from: "func (r EncodedRequest) ByteCount() int                   { return len(r.body) }",
    to: "func (r EncodedRequest) ByteCount() int                   { return len(r.payload) }",
  },
  {
    name: "encoded payload byte count",
    from: "func (r EncodedRequest) PayloadByteCount() int            { return len(r.payload) }",
    to: "func (r EncodedRequest) PayloadByteCount() int            { return len(r.body) }",
  },
  {
    name: "literal UTF-8 JSON representation",
    from: "encoder.SetEscapeHTML(false)",
    to: "encoder.SetEscapeHTML(true)",
  },
  {
    name: "nonempty root identity",
    from: "(s.RootVolume == 0 && s.RootObject == 0)",
    to: "false",
  },
  {
    name: "source-neutral relative path",
    from: '!strings.Contains(value, "\\\\")',
    to: "true",
  },
  {
    name: "single source-file provenance",
    from: "} else if chunk.SourceFile != sourceFile {",
    to: "} else if false {",
  },
  {
    name: "physical cursor excludes revisions",
    from: 'c.PreviousRevisionSHA256 != "" || c.RevisionSHA256 != ""',
    to: "false",
  },
  {
    name: "snapshot cursor must advance",
    from: "c.PreviousRevisionSHA256 == c.RevisionSHA256",
    to: "false",
  },
  {
    name: "empty physical interval",
    from: "c.EndInclusive < c.StartExclusive",
    to: "c.EndInclusive <= c.StartExclusive",
  },
  {
    name: "signed physical cursor range",
    from: "c.StartExclusive > math.MaxInt64 || c.EndInclusive > math.MaxInt64 ||",
    to: "false ||",
  },
  {
    name: "timestamp UTC normalization",
    from: "c.Timestamp.UTC().Format(time.RFC3339Nano)",
    to: "c.Timestamp.Format(time.RFC3339Nano)",
  },
  {
    name: "nonzero timestamp",
    from: "return !value.IsZero() && year >= 1 && year <= 9999",
    to: "return true && year >= 1 && year <= 9999",
  },
  {
    name: "timestamp lower RFC3339 bound",
    from: "return !value.IsZero() && year >= 1 && year <= 9999",
    to: "return !value.IsZero() && true && year <= 9999",
  },
  {
    name: "timestamp upper RFC3339 bound",
    from: "return !value.IsZero() && year >= 1 && year <= 9999",
    to: "return !value.IsZero() && year >= 1 && true",
  },
  {
    name: "unique chunk ids",
    from: "if _, exists := seen[chunk.ID]; exists {",
    to: "if _, exists := seen[chunk.ID]; exists && false {",
  },
  {
    name: "parent cannot be self",
    from: "c.ParentChunkID == c.ID",
    to: "false",
  },
  {
    name: "child cannot be self",
    from: "child == c.ID",
    to: "false",
  },
  {
    name: "unique child ids",
    from: "if _, exists := childSet[child]; exists {",
    to: "if _, exists := childSet[child]; exists && false {",
  },
  {
    name: "closed machine id alphabet",
    from: 'strings.ContainsRune("_.:-", rune(character))',
    to: 'strings.ContainsRune("_.:- ", rune(character))',
  },
  {
    name: "chunk identity machine validation",
    from: "if !validMachineID(identity.MachineID) {",
    to: "if false {",
  },
  {
    name: "chunk identity machine binding",
    from: "MachineID:     identity.MachineID,",
    to: 'MachineID:     "machine-A",',
  },
  {
    name: "chunk identity signed source line",
    from: "identity.SourceLine > math.MaxInt64 || identity.ContentSHA256 == ([sha256.Size]byte{})",
    to: "false || identity.ContentSHA256 == ([sha256.Size]byte{})",
  },
  {
    name: "chunk identity nonzero content digest",
    from: "identity.SourceLine > math.MaxInt64 || identity.ContentSHA256 == ([sha256.Size]byte{})",
    to: "identity.SourceLine > math.MaxInt64 || false",
  },
  {
    name: "chunk signed source line",
    from: "!validTimestamp(c.Timestamp) || !validText(c.SourceFile, 4096, false) || c.SourceLine > math.MaxInt64",
    to: "!validTimestamp(c.Timestamp) || !validText(c.SourceFile, 4096, false) || false",
  },
  {
    name: "nonblank metadata",
    from: 'strings.TrimSpace(value) == "" || len(value) > maximum || !utf8.ValidString(value)',
    to: 'value == "" || len(value) > maximum || !utf8.ValidString(value)',
  },
  {
    name: "metadata line-break closure",
    from: "if allowLineBreaks && (character == '\\n' || character == '\\r' || character == '\\t') {",
    to: "if true && (character == '\\n' || character == '\\r' || character == '\\t') {",
  },
  {
    name: "unicode control closure",
    from: "if !unicode.IsControl(character) {",
    to: "if character >= '\\u0080' || !unicode.IsControl(character) {",
  },
  {
    name: "lowercase digest alphabet",
    from: "(character < 'a' || character > 'f') {",
    to: "(character < 'a' || character > 'f') && (character < 'A' || character > 'F') {",
  },
  {
    name: "inclusive request byte ceiling",
    from: "if len(body) > MaxRequestBytes {",
    to: "if len(body) >= MaxRequestBytes {",
  },
];

for (const mutation of mutations) {
  const directory = mkdtempSync(join(tmpdir(), "history-rag-native-ingest-wire-mutation-"));
  try {
    cpSync(join(root, "go.mod"), join(directory, "go.mod"));
    cpSync(join(root, "go.sum"), join(directory, "go.sum"));
    cpSync(
      join(root, "internal", "history", "ingest"),
      join(directory, "internal", "history", "ingest"),
      { recursive: true },
    );

    const target = join(directory, "internal", "history", "ingest", "encode.go");
    const source = readFileSync(target, "utf8");
    const first = source.indexOf(mutation.from);
    if (first < 0 || source.indexOf(mutation.from, first + mutation.from.length) >= 0) {
      throw new Error(`${mutation.name}: mutation anchor must occur exactly once`);
    }
    writeFileSync(target, source.replace(mutation.from, mutation.to));

    let survived = true;
    try {
      execFileSync("go", ["test", "./internal/history/ingest", "-count=1"], {
        cwd: directory,
        env: { ...process.env, GOTOOLCHAIN: "go1.27.0" },
        stdio: "pipe",
      });
    } catch {
      survived = false;
    }
    if (survived) {
      throw new Error(`${mutation.name}: mutation survived`);
    }
    process.stdout.write(`KILLED ${mutation.name}\n`);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
}

process.stdout.write(`PASS ${mutations.length}/${mutations.length} mutations killed\n`);
