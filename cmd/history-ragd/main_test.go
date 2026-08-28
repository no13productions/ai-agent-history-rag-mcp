package main

import (
	"os"
	"regexp"
	"testing"

	historyprocess "github.com/no13productions/ai-agent-history-rag-mcp/internal/history/process"
)

func TestParseModeIsClosed(t *testing.T) {
	for command, want := range map[string]historyprocess.Mode{"start": historyprocess.Start, "supervise": historyprocess.Supervise} {
		got, err := parseMode(command)
		if err != nil || got != want {
			t.Fatalf("parseMode(%q) = %v, %v", command, got, err)
		}
	}
	for _, command := range []string{"", "stop", "restart", "START", "start "} {
		if _, err := parseMode(command); err == nil {
			t.Fatalf("parseMode(%q) accepted", command)
		}
	}
}

func TestModuleHonorsFleetFloors(t *testing.T) {
	payload, err := os.ReadFile("../../go.mod")
	if err != nil {
		t.Fatal(err)
	}
	checks := map[string]string{
		"Go toolchain": `(?m)^go 1\.27\.0$`,
		"x/sys":        `(?m)^require golang\.org/x/sys v0\.46\.0$`,
	}
	for name, pattern := range checks {
		if !regexp.MustCompile(pattern).Match(payload) {
			t.Fatalf("%s fleet floor missing from go.mod", name)
		}
	}
}
