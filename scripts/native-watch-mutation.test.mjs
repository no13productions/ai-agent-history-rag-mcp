#!/usr/bin/env node

import { execFileSync } from "node:child_process";
import { cpSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");

const mutations = [
  {
    name: "global snapshot ceiling",
    file: "internal/history/watch/root.go",
    from: "maxBytes < 0 || maxBytes > MaxSourceSnapshotBytes",
    to: "maxBytes < 0",
  },
  {
    name: "lexical parent refusal",
    file: "internal/history/watch/root.go",
    from: 'part == "" || part == "." || part == ".."',
    to: 'part == "" || part == "."',
  },
  {
    name: "initial source size bound",
    file: "internal/history/watch/root.go",
    from: "sourceSize < 0 || sourceSize > maxBytes",
    to: "sourceSize < 0",
  },
  {
    name: "snapshot growth refusal",
    file: "internal/history/watch/root.go",
    from: "count != 0 || (readErr != nil && !errors.Is(readErr, io.EOF))",
    to: "false",
  },
  {
    name: "pinned root identity",
    file: "internal/history/watch/root_unix.go",
    from: "!sameUnixIdentity(r.stat, held) || !sameUnixIdentity(held, named)",
    to: "!sameUnixIdentity(r.stat, held)",
  },
  {
    name: "single-link source",
    file: "internal/history/watch/root_unix.go",
    from: "value.Mode&unix.S_IFMT != unix.S_IFREG || value.Nlink != 1",
    to: "value.Mode&unix.S_IFMT != unix.S_IFREG",
  },
  {
    name: "no-follow source open",
    file: "internal/history/watch/root_unix.go",
    from: "unix.O_RDONLY|unix.O_NOFOLLOW|unix.O_CLOEXEC",
    to: "unix.O_RDONLY|unix.O_CLOEXEC",
  },
  {
    name: "post-copy source identity",
    file: "internal/history/watch/root_unix.go",
    from: "!sameUnixIdentity(held, final) || beforeInfo.Size() != afterInfo.Size()",
    to: "beforeInfo.Size() != afterInfo.Size()",
  },
  {
    name: "Windows held parent authority",
    file: "internal/history/watch/root_windows.go",
    from: "RootDirectory: parent",
    to: "RootDirectory: 0",
  },
  {
    name: "Windows single-link source",
    file: "internal/history/watch/root_windows.go",
    from: "information.NumberOfLinks != 1",
    to: "false",
  },
];

for (const mutation of mutations) {
  const directory = mkdtempSync(join(tmpdir(), "history-rag-native-watch-mutation-"));
  try {
    cpSync(join(root, "go.mod"), join(directory, "go.mod"));
    cpSync(join(root, "go.sum"), join(directory, "go.sum"));
    cpSync(join(root, "internal", "history", "watch"), join(directory, "internal", "history", "watch"), { recursive: true });

    const target = join(directory, mutation.file);
    const source = readFileSync(target, "utf8");
    const first = source.indexOf(mutation.from);
    if (first < 0 || source.indexOf(mutation.from, first + mutation.from.length) >= 0) {
      throw new Error(`${mutation.name}: mutation anchor must occur exactly once`);
    }
    writeFileSync(target, source.replace(mutation.from, mutation.to));

    let survived = true;
    try {
      execFileSync("go", ["test", "./internal/history/watch", "-count=1"], {
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
