package main

import (
	"bufio"
	"log/slog"
	"os"
	"path/filepath"
	"strings"

	"ollama-to-openrouter-proxy/handlers"

	"github.com/gin-gonic/gin"
)

var modelFilter []string

// matchesAnyPattern checks if a model name matches any of the glob patterns
func matchesAnyPattern(modelName string, patterns []string) bool {
	if len(patterns) == 0 {
		return true // No filter means all models are allowed
	}

	for _, pattern := range patterns {
		matched, err := filepath.Match(pattern, modelName)
		if err != nil {
			slog.Warn("Invalid glob pattern", "pattern", pattern, "error", err)
			// Fall back to exact string match if pattern is invalid
			if pattern == modelName {
				return true
			}
			continue
		}
		if matched {
			return true
		}
	}
	return false
}

func loadModelFilter(path string) ([]string, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	var filter []string

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line != "" {
			filter = append(filter, line)
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return filter, nil
}

func main() {
	r := gin.Default()

	// Load the API key from environment variables or command-line arguments.
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		if len(os.Args) > 1 {
			apiKey = os.Args[1]
		} else {
			slog.Error("OPENAI_API_KEY environment variable or command-line argument not set.")
			return
		}
	}

	provider := handlers.NewOpenrouterProvider(apiKey)

	filter, err := loadModelFilter("models-filter")
	if err != nil {
		if os.IsNotExist(err) {
			slog.Info("models-filter file not found. Skipping model filtering.")
			modelFilter = []string{}
		} else {
			slog.Error("Error loading models filter", "Error", err)
			return
		}
	} else {
		modelFilter = filter
		slog.Info("Loaded model patterns from filter:")
		for _, pattern := range modelFilter {
			slog.Info(" - " + pattern)
		}
	}

	// Health check endpoints
	r.GET("/", handlers.HealthCheck)
	r.HEAD("/", handlers.HealthCheckHead)
	r.GET("/api/version", handlers.GetVersion)

	// Model endpoints
	r.GET("/api/tags", handlers.GetTags(provider, modelFilter))
	r.POST("/api/show", handlers.ShowModel(provider))
	r.POST("/api/pull", handlers.PullModel(provider))

	// Chat and generate endpoints
	r.POST("/api/chat", handlers.HandleChat(provider))
	r.POST("/api/generate", handlers.HandleGenerate(provider))

	r.Run(":11434")
}
