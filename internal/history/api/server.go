package api

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"sort"
	"time"

	historyauth "github.com/no13productions/ai-agent-history-rag-mcp/internal/history/auth"
)

const (
	MaxRequestBodyBytes              int64 = 4 << 10
	MaxResponseBytes                       = 32 << 10
	MaxHeaderBytes                         = 16 << 10
	requiredReadinessDependencyCount       = 2
)

type Verifier interface {
	VerifyBearer(header string) (historyauth.KeyKind, error)
}

type Readiness interface {
	Name() string
	Ready(context.Context) error
}

type Config struct {
	AuthEnabled bool
}

type readinessDependency struct {
	name      string
	readiness Readiness
}

type Server struct {
	config       Config
	verifier     Verifier
	dependencies []readinessDependency
	handler      http.Handler
}

func New(config Config, verifier Verifier, dependencies []Readiness) (*Server, error) {
	if config.AuthEnabled && verifier == nil {
		return nil, errors.New("authenticated API requires verifier")
	}
	seen := make(map[string]struct{})
	ownedDependencies := make([]readinessDependency, 0, len(dependencies))
	for _, dependency := range dependencies {
		if dependency == nil {
			return nil, errors.New("readiness dependency name required")
		}
		name := dependency.Name()
		if name == "" {
			return nil, errors.New("readiness dependency name required")
		}
		if !isRequiredReadinessDependency(name) {
			return nil, errors.New("unknown readiness dependency")
		}
		if _, exists := seen[name]; exists {
			return nil, errors.New("duplicate readiness dependency")
		}
		seen[name] = struct{}{}
		ownedDependencies = append(ownedDependencies, readinessDependency{name: name, readiness: dependency})
	}
	sort.Slice(ownedDependencies, func(i, j int) bool { return ownedDependencies[i].name < ownedDependencies[j].name })
	server := &Server{config: config, verifier: verifier, dependencies: ownedDependencies}
	server.handler = server.routes()
	return server, nil
}

func isRequiredReadinessDependency(name string) bool {
	switch name {
	case "store", "watcher":
		return true
	default:
		return false
	}
}

func (s *Server) Handler() http.Handler { return s.handler }

func (s *Server) HTTPServer(address string) *http.Server {
	return &http.Server{
		Addr:              address,
		Handler:           s.handler,
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       10 * time.Second,
		WriteTimeout:      10 * time.Second,
		IdleTimeout:       60 * time.Second,
		MaxHeaderBytes:    MaxHeaderBytes,
	}
}

func (s *Server) routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", s.withAuth(s.handleReadiness))
	mux.HandleFunc("GET /status", s.withAuth(s.handleReadiness))
	mux.HandleFunc("POST /api/positions", s.withAuth(s.handleCursorMutation))
	return fixedHeaders(mux)
}

func fixedHeaders(next http.Handler) http.Handler {
	return http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		response.Header().Set("Content-Type", "application/json; charset=utf-8")
		response.Header().Set("Cache-Control", "no-store")
		response.Header().Set("X-Content-Type-Options", "nosniff")
		next.ServeHTTP(response, request)
	})
}

func (s *Server) withAuth(next http.HandlerFunc) http.HandlerFunc {
	return func(response http.ResponseWriter, request *http.Request) {
		if s.config.AuthEnabled {
			header := request.Header.Get("Authorization")
			if len(header) > len("Bearer ")+historyauth.MaxCredentialBytes {
				writeJSON(response, http.StatusUnauthorized, map[string]any{"error": "unauthorized"})
				return
			}
			if _, err := s.verifier.VerifyBearer(header); err != nil {
				writeJSON(response, http.StatusUnauthorized, map[string]any{"error": "unauthorized"})
				return
			}
		}
		next(response, request)
	}
}

func (s *Server) handleReadiness(response http.ResponseWriter, request *http.Request) {
	type result struct {
		Name  string `json:"name"`
		Ready bool   `json:"ready"`
	}
	results := make([]result, 0, len(s.dependencies))
	ready := len(s.dependencies) == requiredReadinessDependencyCount
	for _, dependency := range s.dependencies {
		dependencyReady := dependency.readiness.Ready(request.Context()) == nil
		ready = ready && dependencyReady
		results = append(results, result{Name: dependency.name, Ready: dependencyReady})
	}
	status := "ready"
	code := http.StatusOK
	if !ready {
		status = "not_ready"
		code = http.StatusServiceUnavailable
	}
	writeJSON(response, code, struct {
		Status       string   `json:"status"`
		Dependencies []result `json:"dependencies"`
	}{Status: status, Dependencies: results})
}

func (s *Server) handleCursorMutation(response http.ResponseWriter, request *http.Request) {
	if request.ContentLength > MaxRequestBodyBytes {
		writeJSON(response, http.StatusRequestEntityTooLarge, map[string]any{"error": "request_too_large"})
		return
	}
	body := http.MaxBytesReader(response, request.Body, MaxRequestBodyBytes)
	defer body.Close()
	var discard any
	decoder := json.NewDecoder(body)
	if err := decoder.Decode(&discard); err != nil {
		var maxBytesError *http.MaxBytesError
		if errors.As(err, &maxBytesError) {
			writeJSON(response, http.StatusRequestEntityTooLarge, map[string]any{"error": "request_too_large"})
			return
		}
		writeJSON(response, http.StatusBadRequest, map[string]any{"error": "invalid_request"})
		return
	}
	if decoder.Decode(&discard) == nil {
		writeJSON(response, http.StatusBadRequest, map[string]any{"error": "invalid_request"})
		return
	}
	writeJSON(response, http.StatusConflict, map[string]any{"error": "cursor_sync_forbidden"})
}

func writeJSON(response http.ResponseWriter, status int, value any) {
	payload, err := json.Marshal(value)
	if err != nil || len(payload)+1 > MaxResponseBytes {
		response.WriteHeader(http.StatusInternalServerError)
		_, _ = response.Write([]byte("{\"error\":\"response_unavailable\"}\n"))
		return
	}
	response.WriteHeader(status)
	_, _ = response.Write(append(payload, '\n'))
}
