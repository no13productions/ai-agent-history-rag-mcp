#!/usr/bin/env node

import { execFileSync } from "node:child_process";
import { cpSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");

const requiredModuleFloors = new Map([
  ["cloud.google.com/go/compute/metadata", "v0.9.0"],
  ["golang.org/x/oauth2", "v0.36.0"],
  ["golang.org/x/sys", "v0.46.0"],
]);

function versionParts(version) {
  const match = /^v(\d+)\.(\d+)\.(\d+)$/.exec(version);
  if (!match) {
    throw new Error(`unsupported module version ${version}`);
  }
  return match.slice(1).map(Number);
}

function compareVersions(left, right) {
  const leftParts = versionParts(left);
  const rightParts = versionParts(right);
  for (let index = 0; index < leftParts.length; index += 1) {
    if (leftParts[index] !== rightParts[index]) {
      return leftParts[index] - rightParts[index];
    }
  }
  return 0;
}

function assertFleetDependencyPolicy(directory) {
  const source = readFileSync(join(directory, "go.mod"), "utf8");
  if (!/^go 1\.27\.0$/m.test(source)) {
    throw new Error("go.mod must retain the Go 1.27.0 fleet floor");
  }
  for (const [modulePath, floor] of requiredModuleFloors) {
    const escapedPath = modulePath.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const match = new RegExp(`^\\t${escapedPath} (v\\d+\\.\\d+\\.\\d+)$`, "m").exec(source);
    if (!match || compareVersions(match[1], floor) < 0) {
      throw new Error(`${modulePath} must be declared at or above ${floor}`);
    }
  }
}

const mutations = [
  {
    name: "nested exportable key rejection",
    file: "internal/gcpauth/selector.go",
    replacements: [
      {
        from: 'if key == "private_key" || key == "private_key_id" {',
        to: "if false {",
      },
      {
        from: '[]string{"type", "client_id", "client_secret", "refresh_token", "token_uri", "rapt_token", "universe_domain", "account"}',
        to: '[]string{"type", "client_id", "client_secret", "refresh_token", "token_uri", "rapt_token", "universe_domain", "account", "private_key", "private_key_id"}',
      },
    ],
  },
  {
    name: "authorized-user source constraint",
    file: "internal/gcpauth/selector.go",
    from: 'if sourceType, _ := source["type"].(string); sourceType != "authorized_user" {',
    to: 'if sourceType, _ := source["type"].(string); sourceType == "authorized_user" {',
  },
  {
    name: "static credential environment rejection",
    file: "internal/gcpauth/selector.go",
    from: 'for _, key := range []string{"GOOGLE_APPLICATION_CREDENTIALS", "CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE", "CLOUDSDK_CONFIG"} {\n\t\tif value := strings.TrimSpace(os.Getenv(key)); value != "" {',
    to: 'for _, key := range []string{"GOOGLE_APPLICATION_CREDENTIALS", "CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE", "CLOUDSDK_CONFIG"} {\n\t\tif false {',
  },
  {
    name: "PQC downgrade guard",
    file: "internal/gcpauth/selector.go",
    from: "if strings.Contains(value, fragment) {",
    to: "if false {",
  },
  {
    name: "caller scope closure",
    file: "internal/gcpauth/selector.go",
    from: "if len(scopes) != 1 || scopes[0] != CloudPlatformScope {",
    to: "if false {",
  },
  {
    name: "carrier scope presence",
    file: "internal/gcpauth/selector.go",
    from: 'configuredScopes, exists := raw["scopes"]\n\tif !exists {',
    to: 'configuredScopes, exists := raw["scopes"]\n\tif false {',
  },
  {
    name: "impersonation quota-project binding",
    file: "internal/gcpauth/selector.go",
    from: 'configuredQuotaProject, exists := raw["quota_project_id"]\n\tif !exists {',
    to: 'configuredQuotaProject, exists := raw["quota_project_id"]\n\tif false {',
  },
  {
    name: "Spanner-only production shape",
    file: "internal/runtimeconfig/config.go",
    from: 'ProductionStorageBackend          = "spanner"',
    to: 'ProductionStorageBackend          = "lancedb"',
  },
  {
    name: "loopback status port fence",
    file: "internal/runtimeconfig/config.go",
    from: "if c.StatusServerPort != ProductionStatusServerPort {",
    to: "if false {",
  },
  {
    name: "x/sys fleet floor",
    file: "go.mod",
    from: "golang.org/x/sys v0.46.0",
    to: "golang.org/x/sys v0.45.0",
  },
];

assertFleetDependencyPolicy(root);

for (const mutation of mutations) {
  const directory = mkdtempSync(join(tmpdir(), "history-rag-native-mutation-"));
  try {
    cpSync(join(root, "go.mod"), join(directory, "go.mod"));
    cpSync(join(root, "go.sum"), join(directory, "go.sum"));
    cpSync(join(root, "internal"), join(directory, "internal"), { recursive: true });

    const target = join(directory, mutation.file);
    const source = readFileSync(target, "utf8");
    let mutated = source;
    const replacements = mutation.replacements ?? [{ from: mutation.from, to: mutation.to }];
    for (const replacement of replacements) {
      const first = mutated.indexOf(replacement.from);
      if (first < 0 || mutated.indexOf(replacement.from, first + replacement.from.length) >= 0) {
        throw new Error(`${mutation.name}: mutation anchor must occur exactly once`);
      }
      mutated = mutated.replace(replacement.from, replacement.to);
    }
    writeFileSync(target, mutated);

    let survived = true;
    try {
      assertFleetDependencyPolicy(directory);
      execFileSync("go", ["test", "./internal/..."], {
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
