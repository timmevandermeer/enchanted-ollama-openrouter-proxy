package handlers

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
)

// Model represents a model from the provider
type Model struct {
	Name       string       `json:"name"`
	Model      string       `json:"model,omitempty"`
	ModifiedAt string       `json:"modified_at,omitempty"`
	Size       int64        `json:"size,omitempty"`
	Digest     string       `json:"digest,omitempty"`
	Details    ModelDetails `json:"details,omitempty"`
}

// ModelDetails represents model details
type ModelDetails struct {
	ParentModel       string   `json:"parent_model"`
	Format            string   `json:"format"`
	Family            string   `json:"family"`
	Families          []string `json:"families"`
	ParameterSize     string   `json:"parameter_size"`
	QuantizationLevel string   `json:"quantization_level"`
}

// GetTags handles GET /api/tags endpoint
func GetTags(provider *OpenrouterProvider, modelFilter map[string]struct{}) gin.HandlerFunc {
	return func(c *gin.Context) {
		models, err := provider.GetModels()
		if err != nil {
			slog.Error("Error getting models", "Error", err)
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		// Construct a new array of model objects with extra fields
		newModels := make([]map[string]interface{}, 0, len(models))
		for _, m := range models {
			// If filter is not empty, check if model is in filter
			if len(modelFilter) > 0 {
				if _, ok := modelFilter[m.Model]; !ok {
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
	}
}

// ShowModel handles POST /api/show endpoint
func ShowModel(provider *OpenrouterProvider) gin.HandlerFunc {
	return func(c *gin.Context) {
		var request ShowRequest

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
	}
}

// PullModel handles POST /api/pull endpoint
func PullModel(provider *OpenrouterProvider) gin.HandlerFunc {
	return func(c *gin.Context) {
		var request PullRequest

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
	}
}
