package api

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	historyauth "github.com/no13productions/ai-agent-history-rag-mcp/internal/history/auth"
)

type fakeVerifier struct{ key string }

func (f fakeVerifier) VerifyBearer(header string) (historyauth.KeyKind, error) {
	if header != "Bearer "+f.key {
		return "", historyauth.ErrInvalidCredential
	}
	return historyauth.ActiveKey, nil
}

type dependency struct {
	name string
	err  error
}

func (d dependency) Name() string                { return d.name }
func (d dependency) Ready(context.Context) error { return d.err }

func request(t *testing.T, handler http.Handler, method, path, auth, body string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(method, path, strings.NewReader(body))
	if auth != "" {
		req.Header.Set("Authorization", auth)
	}
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, req)
	return recorder
}

func TestReadinessFailsClosedUntilAllDependenciesAttach(t *testing.T) {
	server, err := New(Config{AuthEnabled: false}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	response := request(t, server.Handler(), http.MethodGet, "/health", "", "")
	if response.Code != http.StatusServiceUnavailable || response.Body.Len() > MaxResponseBytes {
		t.Fatalf("health = %d %q", response.Code, response.Body.String())
	}
	var payload map[string]any
	if err := json.Unmarshal(response.Body.Bytes(), &payload); err != nil {
		t.Fatal(err)
	}
	if payload["status"] != "not_ready" {
		t.Fatalf("status = %#v", payload)
	}

	server, err = New(Config{AuthEnabled: false}, nil, []Readiness{
		dependency{name: "store", err: nil},
		dependency{name: "watcher", err: errors.New("not attached")},
	})
	if err != nil {
		t.Fatal(err)
	}
	response = request(t, server.Handler(), http.MethodGet, "/status", "", "")
	if response.Code != http.StatusServiceUnavailable {
		t.Fatalf("status code = %d", response.Code)
	}

	server, err = New(Config{AuthEnabled: false}, nil, []Readiness{
		dependency{name: "watcher"}, dependency{name: "store"},
	})
	if err != nil {
		t.Fatal(err)
	}
	response = request(t, server.Handler(), http.MethodGet, "/health", "", "")
	if response.Code != http.StatusOK || !strings.Contains(response.Body.String(), `"status":"ready"`) {
		t.Fatalf("ready health = %d %q", response.Code, response.Body.String())
	}
}

func TestReadinessRequiresExactClosedDependencyRoster(t *testing.T) {
	for _, dependencies := range [][]Readiness{
		{dependency{name: "store"}},
		{dependency{name: "watcher"}},
	} {
		server, err := New(Config{AuthEnabled: false}, nil, dependencies)
		if err != nil {
			t.Fatal(err)
		}
		response := request(t, server.Handler(), http.MethodGet, "/health", "", "")
		if response.Code != http.StatusServiceUnavailable || !strings.Contains(response.Body.String(), `"status":"not_ready"`) {
			t.Fatalf("incomplete roster returned %d %q", response.Code, response.Body.String())
		}
	}

	if _, err := New(Config{AuthEnabled: false}, nil, []Readiness{dependency{name: "other"}}); err == nil {
		t.Fatal("unknown readiness dependency was accepted")
	}
	if _, err := New(Config{AuthEnabled: false}, nil, []Readiness{
		dependency{name: "store"}, dependency{name: "store"},
	}); err == nil {
		t.Fatal("duplicate readiness dependency was accepted")
	}
}

func TestOperationalRoutesAuthenticateAndBoundInputs(t *testing.T) {
	server, err := New(Config{AuthEnabled: true}, fakeVerifier{key: "secret"}, []Readiness{dependency{name: "store"}, dependency{name: "watcher"}})
	if err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{"/health", "/status"} {
		if got := request(t, server.Handler(), http.MethodGet, path, "", ""); got.Code != http.StatusUnauthorized {
			t.Fatalf("%s unauthenticated = %d", path, got.Code)
		}
		if got := request(t, server.Handler(), http.MethodGet, path, "Bearer secret", ""); got.Code != http.StatusOK {
			t.Fatalf("%s authenticated = %d %q", path, got.Code, got.Body.String())
		}
	}
	oversizedAuth := "Bearer " + strings.Repeat("x", historyauth.MaxCredentialBytes+1)
	if got := request(t, server.Handler(), http.MethodGet, "/health", oversizedAuth, ""); got.Code != http.StatusUnauthorized {
		t.Fatalf("oversized auth = %d", got.Code)
	}
	body := strings.Repeat("x", int(MaxRequestBodyBytes+1))
	if got := request(t, server.Handler(), http.MethodPost, "/api/positions", "Bearer secret", body); got.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("oversized body = %d", got.Code)
	}
	chunked := httptest.NewRequest(http.MethodPost, "/api/positions", strings.NewReader(`{"value":"`+strings.Repeat("x", int(MaxRequestBodyBytes))+`"}`))
	chunked.ContentLength = -1
	chunked.Header.Set("Authorization", "Bearer secret")
	chunkedResponse := httptest.NewRecorder()
	server.Handler().ServeHTTP(chunkedResponse, chunked)
	if chunkedResponse.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("chunked oversized body = %d %q", chunkedResponse.Code, chunkedResponse.Body.String())
	}
}

func TestDirectCursorMutationAlwaysRefused(t *testing.T) {
	server, err := New(Config{AuthEnabled: true}, fakeVerifier{key: "secret"}, []Readiness{dependency{name: "store"}, dependency{name: "watcher"}})
	if err != nil {
		t.Fatal(err)
	}
	response := request(t, server.Handler(), http.MethodPost, "/api/positions", "Bearer secret", `{}`)
	if response.Code != http.StatusConflict || response.Body.String() != "{\"error\":\"cursor_sync_forbidden\"}\n" {
		t.Fatalf("cursor route = %d %q", response.Code, response.Body.String())
	}
}

func TestHTTPServerTimeoutsAndBoundsAreFixed(t *testing.T) {
	server, err := New(Config{AuthEnabled: false}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	httpServer := server.HTTPServer("127.0.0.1:4680")
	if httpServer.Addr != "127.0.0.1:4680" || httpServer.ReadHeaderTimeout <= 0 || httpServer.ReadTimeout <= 0 || httpServer.WriteTimeout <= 0 || httpServer.IdleTimeout <= 0 || httpServer.MaxHeaderBytes != MaxHeaderBytes {
		t.Fatalf("unsafe HTTP server: %#v", httpServer)
	}
}
