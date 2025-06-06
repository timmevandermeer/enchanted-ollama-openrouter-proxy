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
	openai "github.com/sashabaranov/go-openai"
)

// HandleChat handles POST /api/chat endpoint
func HandleChat(provider *OpenrouterProvider) gin.HandlerFunc {
	return func(c *gin.Context) {
		var request ChatRequest

		// Parse the JSON request
		if err := c.ShouldBindJSON(&request); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid JSON payload"})
			return
		}

		// Convert Ollama messages to OpenAI format
		openaiMessages := ConvertOllamaMessagesToOpenAI(request.Messages)

		// Log if we're processing any images
		imageCount := 0
		for _, msg := range request.Messages {
			imageCount += len(msg.Images)
		}
		if imageCount > 0 {
			slog.Info("Processing multimodal request", "imageCount", imageCount)
		}

		// Determine if streaming is requested (default to true for /api/chat)
		streamRequested := true
		if request.Stream != nil {
			streamRequested = *request.Stream
		}

		if !streamRequested {
			handleNonStreamingChat(c, provider, request, openaiMessages)
			return
		}

		handleStreamingChat(c, provider, request, openaiMessages)
	}
}

func handleNonStreamingChat(c *gin.Context, provider *OpenrouterProvider, request ChatRequest, openaiMessages []openai.ChatCompletionMessage) {
	fullModelName, err := provider.GetFullModelName(request.Model)
	if err != nil {
		slog.Error("Error getting full model name", "Error", err)
		c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
		return
	}

	// Call Chat to get the complete response
	response, err := provider.Chat(openaiMessages, fullModelName)
	if err != nil {
		slog.Error("Failed to get chat response", "Error", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	// Format the response according to Ollama's format
	if len(response.Choices) == 0 {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "No response from model"})
		return
	}

	// Extract the content from the response
	content := ""
	if len(response.Choices) > 0 && response.Choices[0].Message.Content != "" {
		content = response.Choices[0].Message.Content
	}

	// Get finish reason, default to "stop" if not provided
	finishReason := "stop"
	if response.Choices[0].FinishReason != "" {
		finishReason = string(response.Choices[0].FinishReason)
	}

	// Create Ollama-compatible response
	ollamaResponse := map[string]interface{}{
		"model":      fullModelName,
		"created_at": time.Now().Format(time.RFC3339),
		"message": map[string]string{
			"role":    "assistant",
			"content": content,
		},
		"done":              true,
		"finish_reason":     finishReason,
		"total_duration":    response.Usage.TotalTokens * 10, // Approximate duration based on token count
		"load_duration":     0,
		"prompt_eval_count": response.Usage.PromptTokens,
		"eval_count":        response.Usage.CompletionTokens,
		"eval_duration":     response.Usage.CompletionTokens * 10, // Approximate duration based on token count
	}

	c.JSON(http.StatusOK, ollamaResponse)
}

func handleStreamingChat(c *gin.Context, provider *OpenrouterProvider, request ChatRequest, openaiMessages []openai.ChatCompletionMessage) {
	slog.Info("Requested model", "model", request.Model)
	fullModelName, err := provider.GetFullModelName(request.Model)
	if err != nil {
		slog.Error("Error getting full model name", "Error", err, "model", request.Model)
		c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
		return
	}
	slog.Info("Using model", "fullModelName", fullModelName)

	// Call ChatStream to get the stream
	stream, err := provider.ChatStream(openaiMessages, fullModelName)
	if err != nil {
		slog.Error("Failed to create stream", "Error", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	defer stream.Close() // Ensure stream closure

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
			// End of stream from the backend provider
			break
		}
		if err != nil {
			slog.Error("Backend stream error", "Error", err)
			errorMsg := map[string]string{"error": "Stream error: " + err.Error()}
			errorJson, _ := json.Marshal(errorMsg)
			fmt.Fprintf(w, "%s\n", string(errorJson))
			flusher.Flush()
			return
		}

		// Save finish reason if present in chunk
		if len(response.Choices) > 0 && response.Choices[0].FinishReason != "" {
			lastFinishReason = string(response.Choices[0].FinishReason)
		}

		// Build JSON response structure for intermediate chunks (Ollama chat format)
		responseJSON := map[string]interface{}{
			"model":      fullModelName,
			"created_at": time.Now().Format(time.RFC3339),
			"message": map[string]string{
				"role":    "assistant",
				"content": response.Choices[0].Delta.Content,
			},
			"done": false, // Always false for intermediate chunks
		}

		// Marshal JSON
		jsonData, err := json.Marshal(responseJSON)
		if err != nil {
			slog.Error("Error marshaling intermediate response JSON", "Error", err)
			return
		}

		// Send JSON object followed by a newline
		fmt.Fprintf(w, "%s\n", string(jsonData))

		// Flush data to send it immediately
		flusher.Flush()
	}

	// Send final message (done: true) in Ollama style
	if lastFinishReason == "" {
		lastFinishReason = "stop"
	}

	finalResponse := map[string]interface{}{
		"model":      fullModelName,
		"created_at": time.Now().Format(time.RFC3339),
		"message": map[string]string{
			"role":    "assistant",
			"content": "", // Empty content for final message
		},
		"done":              true,
		"finish_reason":     lastFinishReason,
		"total_duration":    0,
		"load_duration":     0,
		"prompt_eval_count": 0,
		"eval_count":        0,
		"eval_duration":     0,
	}

	finalJsonData, err := json.Marshal(finalResponse)
	if err != nil {
		slog.Error("Error marshaling final response JSON", "Error", err)
		return
	}

	// Send final JSON object + newline
	fmt.Fprintf(w, "%s\n", string(finalJsonData))
	flusher.Flush()
}
