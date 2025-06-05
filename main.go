package main

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	openai "github.com/sashabaranov/go-openai"
)

// OllamaMessage represents the Ollama message format which may include images
type OllamaMessage struct {
	Role    string   `json:"role"`
	Content string   `json:"content"`
	Images  []string `json:"images,omitempty"` // Base64 encoded images
}

// convertOllamaMessagesToOpenAI converts Ollama-format messages to OpenAI format
func convertOllamaMessagesToOpenAI(ollamaMessages []OllamaMessage) []openai.ChatCompletionMessage {
	var openaiMessages []openai.ChatCompletionMessage

	for _, ollamaMsg := range ollamaMessages {
		openaiMsg := openai.ChatCompletionMessage{
			Role: ollamaMsg.Role,
		}

		// If there are no images, use simple content string
		if len(ollamaMsg.Images) == 0 {
			openaiMsg.Content = ollamaMsg.Content
		} else {
			// Use MultiContent for messages with images
			var parts []openai.ChatMessagePart

			// Add text content if present
			if ollamaMsg.Content != "" {
				parts = append(parts, openai.ChatMessagePart{
					Type: openai.ChatMessagePartTypeText,
					Text: ollamaMsg.Content,
				})
			}

			// Add image parts
			for _, imageData := range ollamaMsg.Images {
				// Ensure the image data has the proper data URL format
				imageURL := imageData
				if !strings.HasPrefix(imageData, "data:") {
					// Assume it's base64 encoded image, add data URL prefix
					imageURL = "data:image/jpeg;base64," + imageData
				}

				parts = append(parts, openai.ChatMessagePart{
					Type: openai.ChatMessagePartTypeImageURL,
					ImageURL: &openai.ChatMessageImageURL{
						URL:    imageURL,
						Detail: openai.ImageURLDetailAuto,
					},
				})
			}

			openaiMsg.MultiContent = parts
		}

		openaiMessages = append(openaiMessages, openaiMsg)
	}

	return openaiMessages
}

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

	provider := NewOpenrouterProvider(apiKey)

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

	r.GET("/", func(c *gin.Context) {
		c.String(http.StatusOK, "Ollama is running")
	})

	r.HEAD("/", func(c *gin.Context) {
		c.String(http.StatusOK, "")
	})

	r.GET("/api/version", func(c *gin.Context) {
		// Return Ollama-compatible version information
		versionInfo := map[string]string{
			"version": "0.1.0-proxy", // Proxy version
		}
		c.JSON(http.StatusOK, versionInfo)
	})

	r.GET("/api/tags", func(c *gin.Context) {
		models, err := provider.GetModels()
		if err != nil {
			slog.Error("Error getting models", "Error", err)
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		filter := modelFilter
		// Construct a new array of model objects with extra fields
		newModels := make([]map[string]interface{}, 0, len(models))
		for _, m := range models {
			// Если фильтр пустой, значит пропускаем проверку и берём все модели
			if len(filter) > 0 {
				if _, ok := filter[m.Model]; !ok {
					continue
				}
			}
			newModels = append(newModels, map[string]interface{}{
				"name":        m.Name,
				"model":       m.Model,
				"modified_at": m.ModifiedAt,
				"size":        270898672,
				"digest":      "9077fe9d2ae1a4a41a868836b56b8163731a8fe16621397028c2c76f838c6907",
				"details":     m.Details,
			})
		}

		c.JSON(http.StatusOK, gin.H{"models": newModels})
	})

	r.POST("/api/show", func(c *gin.Context) {
		var request struct {
			Name  string `json:"name,omitempty"`
			Model string `json:"model,omitempty"`
		}

		if err := c.ShouldBindJSON(&request); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid JSON payload"})
			return
		}

		modelName := request.Name
		if modelName == "" {
			modelName = request.Model // Fallback to model field if name is empty
		}

		if modelName == "" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Model name is required"})
			return
		}

		details, err := provider.GetModelDetails(modelName)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		c.JSON(http.StatusOK, details)
	})

	r.POST("/api/pull", func(c *gin.Context) {
		var request struct {
			Name     string `json:"name"`
			Model    string `json:"model,omitempty"` // Optional, for compatibility
			Insecure bool   `json:"insecure,omitempty"`
			Stream   *bool  `json:"stream,omitempty"`
		}

		// Parse the JSON request
		if err := c.ShouldBindJSON(&request); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid JSON payload"})
			return
		}

		var modelName = request.Name
		if modelName == "" {
			modelName = request.Model // Fallback to model field if name is empty
		}

		// Validate required fields
		if modelName == "" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Model name is required"})
			return
		}

		// Determine if streaming is requested (default to true for /api/pull)
		streamRequested := true
		if request.Stream != nil {
			streamRequested = *request.Stream
		}

		slog.Info("Pull request", "model", modelName, "stream", streamRequested)

		// Check if the model exists in available models
		_, err := provider.GetFullModelName(modelName)
		if err != nil {
			slog.Error("Model not found", "Error", err, "model", modelName)
			c.JSON(http.StatusNotFound, gin.H{"error": "Model not found: " + modelName})
			return
		}

		if !streamRequested {
			// Handle non-streaming response
			pullResponse := map[string]interface{}{
				"status":    "success",
				"digest":    "sha256:abc123def456...", // Mock digest
				"total":     1000000,                  // Mock total size
				"completed": 1000000,                  // Mock completed size
			}
			c.JSON(http.StatusOK, pullResponse)
			return
		}

		// Handle streaming response
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

		// Simulate pulling progress with multiple status updates
		statuses := []map[string]interface{}{
			{
				"status": "pulling manifest",
			},
			{
				"status": "pulling model",
				"digest": "sha256:abc123def456...",
				"total":  1000000,
			},
			{
				"status":    "downloading",
				"digest":    "sha256:abc123def456...",
				"total":     1000000,
				"completed": 250000,
			},
			{
				"status":    "downloading",
				"digest":    "sha256:abc123def456...",
				"total":     1000000,
				"completed": 500000,
			},
			{
				"status":    "downloading",
				"digest":    "sha256:abc123def456...",
				"total":     1000000,
				"completed": 750000,
			},
			{
				"status":    "downloading",
				"digest":    "sha256:abc123def456...",
				"total":     1000000,
				"completed": 1000000,
			},
			{
				"status": "verifying sha256 digest",
			},
			{
				"status": "writing manifest",
			},
			{
				"status": "removing any unused layers",
			},
			{
				"status": "success",
			},
		}

		// Send each status update with a small delay to simulate real pulling
		for _, status := range statuses {
			jsonData, err := json.Marshal(status)
			if err != nil {
				slog.Error("Error marshaling pull status JSON", "Error", err)
				return
			}

			// Log raw JSON for the pull command
			slog.Info("Pull status update", "status_json", string(jsonData))

			fmt.Fprintf(w, "%s\n", string(jsonData))
			flusher.Flush()

			// Small delay to simulate progress (remove in production if too slow)
			time.Sleep(100 * time.Millisecond)
		}
	})

	r.POST("/api/chat", func(c *gin.Context) {
		var request struct {
			Model    string          `json:"model"`
			Messages []OllamaMessage `json:"messages"`
			Stream   *bool           `json:"stream"`
		}

		// Parse the JSON request
		if err := c.ShouldBindJSON(&request); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid JSON payload"})
			return
		}

		// Convert Ollama messages to OpenAI format
		openaiMessages := convertOllamaMessagesToOpenAI(request.Messages)
		
		// Log if we're processing any images
		imageCount := 0
		for _, msg := range request.Messages {
			imageCount += len(msg.Images)
		}
		if imageCount > 0 {
			slog.Info("Processing multimodal request", "imageCount", imageCount)
		}

		// Определяем, нужен ли стриминг (по умолчанию true, если не указано для /api/chat)
		// ВАЖНО: Open WebUI может НЕ передавать "stream": true для /api/chat, подразумевая это.
		// Нужно проверить, какой запрос шлет Open WebUI. Если не шлет, ставим true.
		streamRequested := true
		if request.Stream != nil {
			streamRequested = *request.Stream
		}

		// Если стриминг не запрошен, нужно будет реализовать отдельную логику
		// для сбора полного ответа и отправки его одним JSON.
		// Пока реализуем только стриминг.
		if !streamRequested {
			// Handle non-streaming response
			fullModelName, err := provider.GetFullModelName(request.Model)
			if err != nil {
				slog.Error("Error getting full model name", "Error", err)
				// Ollama returns 404 for invalid model names
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
			return
		}

		slog.Info("Requested model", "model", request.Model)
		fullModelName, err := provider.GetFullModelName(request.Model)
		if err != nil {
			slog.Error("Error getting full model name", "Error", err, "model", request.Model)
			// Ollama возвращает 404 на неправильное имя модели
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

		// --- ИСПРАВЛЕНИЯ для NDJSON (Ollama-style) ---

		// Set headers CORRECTLY for Newline Delimited JSON
		c.Writer.Header().Set("Content-Type", "application/x-ndjson") // <--- КЛЮЧЕВОЕ ИЗМЕНЕНИЕ
		c.Writer.Header().Set("Cache-Control", "no-cache")
		c.Writer.Header().Set("Connection", "keep-alive")
		// Transfer-Encoding: chunked устанавливается Gin автоматически

		w := c.Writer // Получаем ResponseWriter
		flusher, ok := w.(http.Flusher)
		if !ok {
			slog.Error("Expected http.ResponseWriter to be an http.Flusher")
			// Отправить ошибку клиенту уже сложно, т.к. заголовки могли уйти
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
				// Попытка отправить ошибку в формате NDJSON
				// Ollama обычно просто обрывает соединение или шлет 500 перед этим
				errorMsg := map[string]string{"error": "Stream error: " + err.Error()}
				errorJson, _ := json.Marshal(errorMsg)
				fmt.Fprintf(w, "%s\n", string(errorJson)) // Отправляем ошибку + \n
				flusher.Flush()
				return
			}

			// Сохраняем причину остановки, если она есть в чанке
			if len(response.Choices) > 0 && response.Choices[0].FinishReason != "" {
				lastFinishReason = string(response.Choices[0].FinishReason)
			}

			// Build JSON response structure for intermediate chunks (Ollama chat format)
			responseJSON := map[string]interface{}{
				"model":      fullModelName,
				"created_at": time.Now().Format(time.RFC3339),
				"message": map[string]string{
					"role":    "assistant",
					"content": response.Choices[0].Delta.Content, // Может быть ""
				},
				"done": false, // Всегда false для промежуточных чанков
			}

			// Marshal JSON
			jsonData, err := json.Marshal(responseJSON)
			if err != nil {
				slog.Error("Error marshaling intermediate response JSON", "Error", err)
				return // Прерываем, так как не можем отправить данные
			}

			// Send JSON object followed by a newline
			fmt.Fprintf(w, "%s\n", string(jsonData)) // <--- ИЗМЕНЕНО: Формат NDJSON (JSON + \n)

			// Flush data to send it immediately
			flusher.Flush()
		}

		// --- Отправка финального сообщения (done: true) в стиле Ollama ---

		// Определяем причину остановки (если бэкенд не дал, ставим 'stop')
		// Ollama использует 'stop', 'length', 'content_filter', 'tool_calls'
		if lastFinishReason == "" {
			lastFinishReason = "stop"
		}

		// ВАЖНО: Замените nil на 0 для числовых полей статистики
		finalResponse := map[string]interface{}{
			"model":      fullModelName,
			"created_at": time.Now().Format(time.RFC3339),
			"message": map[string]string{
				"role":    "assistant",
				"content": "", // Пустой контент для финального сообщения
			},
			"done":              true,
			"finish_reason":     lastFinishReason, // Необязательно для /api/chat Ollama, но не вредит
			"total_duration":    0,
			"load_duration":     0,
			"prompt_eval_count": 0, // <--- ИЗМЕНЕНО: nil заменен на 0
			"eval_count":        0, // <--- ИЗМЕНЕНО: nil заменен на 0
			"eval_duration":     0,
		}

		finalJsonData, err := json.Marshal(finalResponse)
		if err != nil {
			slog.Error("Error marshaling final response JSON", "Error", err)
			return
		}

		// Отправляем финальный JSON-объект + newline
		fmt.Fprintf(w, "%s\n", string(finalJsonData)) // <--- ИЗМЕНЕНО: Формат NDJSON
		flusher.Flush()

		// ВАЖНО: Для NDJSON НЕТ 'data: [DONE]' маркера.
		// Клиент понимает конец потока по получению объекта с "done": true
		// и/или по закрытию соединения сервером (что Gin сделает автоматически после выхода из хендлера).

		// --- Конец исправлений ---
	})

	r.POST("/api/generate", func(c *gin.Context) {
		var request struct {
			Model     string                 `json:"model"`
			Prompt    string                 `json:"prompt"`
			Stream    *bool                  `json:"stream"`
			Options   map[string]interface{} `json:"options"`
			Format    string                 `json:"format"`
			Raw       bool                   `json:"raw"`
			Context   []int                  `json:"context"`
			KeepAlive string                 `json:"keep_alive"`
		}

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
			// Handle non-streaming response
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
			return
		}

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
	})

	r.Run(":11434")
}
