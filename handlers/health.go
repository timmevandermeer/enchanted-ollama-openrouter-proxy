package handlers

import (
	"net/http"

	"github.com/gin-gonic/gin"
)

// HealthCheck handles GET / endpoint
func HealthCheck(c *gin.Context) {
	c.String(http.StatusOK, "Ollama is running")
}

// HealthCheckHead handles HEAD / endpoint
func HealthCheckHead(c *gin.Context) {
	c.String(http.StatusOK, "")
}

// GetVersion handles GET /api/version endpoint
func GetVersion(c *gin.Context) {
	// Return Ollama-compatible version information
	versionInfo := map[string]string{
		"version": "0.1.0-proxy", // Proxy version
	}
	c.JSON(http.StatusOK, versionInfo)
}
