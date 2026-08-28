#!/usr/bin/env node

import { execFileSync } from "node:child_process";
import { cpSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");

const mutations = [
  {
    name: "exact vector dimension",
    file: "internal/history/store/plans.go",
    from: "if len(vector) != VectorDimension {",
    to: "if false {",
  },
  {
    name: "finite vector values",
    file: "internal/history/store/plans.go",
    from: "if !isFinite(value) {",
    to: "if false {",
  },
  {
    name: "nonzero vector",
    file: "internal/history/store/plans.go",
    from: "if nonZero == false {",
    to: "if false {",
  },
  {
    name: "mixed batch refusal",
    file: "internal/history/store/plans.go",
    from: "if embedded != 0 && unembedded != 0 {",
    to: "if false {",
  },
  {
    name: "NULL vector search exclusion",
    file: "internal/history/store/plans.go",
    from: 'filters = append(filters, "Vector IS NOT NULL")',
    to: 'filters = append(filters, "Vector IS NULL")',
  },
  {
    name: "machine-scoped deletion predicate",
    file: "internal/history/store/store.go",
    from: 'SQL:    "DELETE FROM ConversationChunks WHERE MachineId = @machine_id",',
    to: 'SQL:    "DELETE FROM ConversationChunks WHERE TRUE",',
  },
  {
    name: "ANN stored-filter coverage",
    file: "internal/history/store/plans.go",
    from: "if hasUnstoredFilters(query.Filter) {",
    to: "if false {",
  },
  {
    name: "bounded literal LIMIT",
    file: "internal/history/store/plans.go",
    from: "if limit < 1 || limit > maximum {",
    to: "if false {",
  },
  {
    name: "256 lowercase deterministic shards",
    file: "internal/history/store/backfill.go",
    from: 'shards[value] = fmt.Sprintf("%02x", value)',
    to: 'shards[value] = fmt.Sprintf("%02X", value)',
  },
  {
    name: "context cancellation before executor",
    file: "internal/history/store/store.go",
    from: "return ctx.Err()",
    to: "return nil",
  },
];

for (const mutation of mutations) {
  const directory = mkdtempSync(join(tmpdir(), "history-rag-native-store-mutation-"));
  try {
    cpSync(join(root, "go.mod"), join(directory, "go.mod"));
    cpSync(join(root, "go.sum"), join(directory, "go.sum"));
    cpSync(join(root, "internal", "history", "store"), join(directory, "internal", "history", "store"), { recursive: true });

    const target = join(directory, mutation.file);
    const source = readFileSync(target, "utf8");
    const first = source.indexOf(mutation.from);
    if (first < 0 || source.indexOf(mutation.from, first + mutation.from.length) >= 0) {
      throw new Error(`${mutation.name}: mutation anchor must occur exactly once`);
    }
    writeFileSync(target, source.replace(mutation.from, mutation.to));

    let survived = true;
    try {
      execFileSync("go", ["test", "./internal/history/store", "-count=1"], {
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
