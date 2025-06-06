package handlers

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
)

// HandleGenerate handles POST /api/generate endpoint
func HandleGenerate(provider *OpenrouterProvider) gin.HandlerFunc {
	return func(c *gin.Context) {
		var request GenerateRequest

		// Parse the JSON request
		if err := c.ShouldBindJSON(&request); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid JSON payload"})
			return
		}

		// Validate required fields
		if request.Model == "" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Model is required"})
			return
		}
		if request.Prompt == "" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Prompt is required"})
			return
		}

		// Determine if streaming is requested (default to false for /api/generate)
		streamRequested := false
		if request.Stream != nil {
			streamRequested = *request.Stream
		}

		slog.Info("Generate request", "model", request.Model, "stream", streamRequested)
		fullModelName, err := provider.GetFullModelName(request.Model)
		if err != nil {
			slog.Error("Error getting full model name", "Error", err, "model", request.Model)
			c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
			return
		}
		slog.Info("Using model for generation", "fullModelName", fullModelName)

		if !streamRequested {
			handleNonStreamingGenerate(c, provider, request, fullModelName)
			return
		}

		handleStreamingGenerate(c, provider, request, fullModelName)
	}
}

func handleNonStreamingGenerate(c *gin.Context, provider *OpenrouterProvider, request GenerateRequest, fullModelName string) {
	response, err := provider.Generate(request.Prompt, fullModelName, request.Options)
	if err != nil {
		slog.Error("Failed to get generation response", "Error", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	// Format the response according to Ollama's /api/generate format
	if len(response.Choices) == 0 {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "No response from model"})
		return
	}

	// Extract the content from the response
	content := ""
	if len(response.Choices) > 0 {
		content = response.Choices[0].Text
	}

	// Create Ollama-compatible response for /api/generate
	ollamaResponse := map[string]interface{}{
		"model":             fullModelName,
		"created_at":        time.Now().Format(time.RFC3339),
		"response":          content,
		"done":              true,
		"context":           []int{}, // Context handling not implemented
		"total_duration":    response.Usage.TotalTokens * 10,
		"load_duration":     0,
		"prompt_eval_count": response.Usage.PromptTokens,
		"eval_count":        response.Usage.CompletionTokens,
		"eval_duration":     response.Usage.CompletionTokens * 10,
	}

	c.JSON(http.StatusOK, ollamaResponse)
}

func handleStreamingGenerate(c *gin.Context, provider *OpenrouterProvider, request GenerateRequest, fullModelName string) {
	// Handle streaming response
	stream, err := provider.GenerateStream(request.Prompt, fullModelName, request.Options)
	if err != nil {
		slog.Error("Failed to create generation stream", "Error", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	defer stream.Close()

	// Set headers for NDJSON streaming
	c.Writer.Header().Set("Content-Type", "application/x-ndjson")
	c.Writer.Header().Set("Cache-Control", "no-cache")
	c.Writer.Header().Set("Connection", "keep-alive")

	w := c.Writer
	flusher, ok := w.(http.Flusher)
	if !ok {
		slog.Error("Expected http.ResponseWriter to be an http.Flusher")
		return
	}

	var lastFinishReason string

	// Stream responses back to the client
	for {
		response, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			slog.Error("Backend generation stream error", "Error", err)
			errorMsg := map[string]string{"error": "Stream error: " + err.Error()}
			errorJson, _ := json.Marshal(errorMsg)
			fmt.Fprintf(w, "%s\n", string(errorJson))
			flusher.Flush()
			return
		}

		// Save finish reason if present
		if len(response.Choices) > 0 && response.Choices[0].FinishReason != "" {
			lastFinishReason = string(response.Choices[0].FinishReason)
		}

		// Build JSON response structure for intermediate chunks
		responseJSON := map[string]interface{}{
			"model":      fullModelName,
			"created_at": time.Now().Format(time.RFC3339),
			"response":   response.Choices[0].Text, // Note: using Text for completions, not Delta.Content
			"done":       false,
		}

		// Marshal and send JSON
		jsonData, err := json.Marshal(responseJSON)
		if err != nil {
			slog.Error("Error marshaling intermediate generation response JSON", "Error", err)
			return
		}

		fmt.Fprintf(w, "%s\n", string(jsonData))
		flusher.Flush()
	}

	// Send final message with done: true
	if lastFinishReason == "" {
		lastFinishReason = "stop"
	}

	finalResponse := map[string]interface{}{
		"model":             fullModelName,
		"created_at":        time.Now().Format(time.RFC3339),
		"response":          "",
		"done":              true,
		"context":           []int{},
		"total_duration":    0,
		"load_duration":     0,
		"prompt_eval_count": 0,
		"eval_count":        0,
		"eval_duration":     0,
	}

	finalJsonData, err := json.Marshal(finalResponse)
	if err != nil {
		slog.Error("Error marshaling final generation response JSON", "Error", err)
		return
	}

	fmt.Fprintf(w, "%s\n", string(finalJsonData))
	flusher.Flush()
}
