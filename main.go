package main

import (
	"bufio"
	"log/slog"
	"os"
	"strings"

	"ollama-to-openrouter-proxy/handlers"

	"github.com/gin-gonic/gin"
)

var modelFilter map[string]struct{}

func loadModelFilter(path string) (map[string]struct{}, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	filter := make(map[string]struct{})

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line != "" {
			filter[line] = struct{}{}
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
			modelFilter = make(map[string]struct{})
		} else {
			slog.Error("Error loading models filter", "Error", err)
			return
		}
	} else {
		modelFilter = filter
		slog.Info("Loaded models from filter:")
		for model := range modelFilter {
			slog.Info(" - " + model)
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
