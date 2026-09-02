package main

import (
	"context"
	"crypto/rand"
	"errors"
	"flag"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"github.com/no13productions/ai-agent-history-rag-mcp/internal/history/api"
	"github.com/no13productions/ai-agent-history-rag-mcp/internal/history/auth"
	"github.com/no13productions/ai-agent-history-rag-mcp/internal/history/config"
	"github.com/no13productions/ai-agent-history-rag-mcp/internal/history/durable"
	historyprocess "github.com/no13productions/ai-agent-history-rag-mcp/internal/history/process"
)

const pskEnvironment = "CLAUDE_HISTORY_RAG_SERVER_PSK"

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, "history-ragd: operation failed")
		os.Exit(1)
	}
}

func run(arguments []string) error {
	if len(arguments) == 0 {
		return errors.New("command required")
	}
	mode, err := parseMode(arguments[0])
	if err != nil {
		return err
	}
	flags := flag.NewFlagSet("history-ragd", flag.ContinueOnError)
	flags.SetOutput(ioDiscard{})
	configPath := flags.String("config", "", "absolute path to the native daemon configuration")
	if err := flags.Parse(arguments[1:]); err != nil || flags.NArg() != 0 || *configPath == "" {
		return errors.New("exactly one --config path is required")
	}
	cfg, err := config.Load(*configPath)
	if err != nil {
		return err
	}
	root, err := durable.OpenRoot(cfg.StateDir)
	if err != nil {
		return err
	}
	defer root.Close()

	var verifier api.Verifier
	if cfg.AuthEnabled {
		secret := os.Getenv(pskEnvironment)
		if secret == "" {
			return errors.New("authentication credential unavailable")
		}
		manager, err := auth.NewManager(root, filepath.Base(cfg.AuthStateFile), rand.Reader)
		if err != nil {
			return err
		}
		if err := manager.Initialize(secret); err != nil {
			return err
		}
		verifier = manager
	}

	ops := historyprocess.NewSystemOps()
	current, err := historyprocess.CurrentRecord(ops, cfg.Executable, cfg.CheckoutRoot)
	if err != nil {
		return err
	}
	controller, err := historyprocess.NewController(root, filepath.Base(cfg.PIDFile), ops, historyprocess.DefaultSleep)
	if err != nil {
		return err
	}
	shouldRun, err := controller.Prepare(context.Background(), mode, current, 15*time.Second, 5*time.Second)
	if err != nil {
		return err
	}
	if !shouldRun {
		fmt.Fprintln(os.Stdout, "history-ragd is already running")
		return nil
	}

	listener, err := net.Listen("tcp", cfg.Listen)
	if err != nil {
		return err
	}
	defer listener.Close()
	if err := controller.WriteRecord(current); err != nil {
		return err
	}
	defer controller.RemoveRecord(current)

	// Watcher and store readiness dependencies are intentionally absent in this
	// foundation phase. The server therefore serves deterministic 503 not_ready
	// responses until a later phase injects both real components.
	apiServer, err := api.New(api.Config{AuthEnabled: cfg.AuthEnabled}, verifier, nil)
	if err != nil {
		return err
	}
	httpServer := apiServer.HTTPServer(cfg.Listen)
	serveResult := make(chan error, 1)
	go func() {
		serveResult <- httpServer.Serve(listener)
	}()

	shutdownSignal := make(chan os.Signal, 1)
	signal.Notify(shutdownSignal, syscall.SIGINT, syscall.SIGTERM)
	defer signal.Stop(shutdownSignal)
	select {
	case err := <-serveResult:
		if !errors.Is(err, http.ErrServerClosed) {
			return err
		}
		return nil
	case <-shutdownSignal:
		ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()
		if err := httpServer.Shutdown(ctx); err != nil {
			return err
		}
		if err := <-serveResult; !errors.Is(err, http.ErrServerClosed) {
			return err
		}
		return nil
	}
}

func parseMode(command string) (historyprocess.Mode, error) {
	switch command {
	case "start":
		return historyprocess.Start, nil
	case "supervise":
		return historyprocess.Supervise, nil
	default:
		return 0, errors.New("command must be start or supervise")
	}
}

type ioDiscard struct{}

func (ioDiscard) Write(payload []byte) (int, error) { return len(payload), nil }
