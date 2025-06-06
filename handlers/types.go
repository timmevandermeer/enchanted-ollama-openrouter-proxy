package handlers

import (
	"github.com/sashabaranov/go-openai"
)

// OllamaMessage represents the Ollama message format which may include images
type OllamaMessage struct {
	Role    string   `json:"role"`
	Content string   `json:"content"`
	Images  []string `json:"images,omitempty"` // Base64 encoded images
}

// ShowRequest represents the request for /api/show endpoint
type ShowRequest struct {
	Name  string `json:"name,omitempty"`
	Model string `json:"model,omitempty"`
}

// PullRequest represents the request for /api/pull endpoint
type PullRequest struct {
	Name     string `json:"name"`
	Model    string `json:"model,omitempty"` // Optional, for compatibility
	Insecure bool   `json:"insecure,omitempty"`
	Stream   *bool  `json:"stream,omitempty"`
}

// ChatRequest represents the request for /api/chat endpoint
type ChatRequest struct {
	Model    string          `json:"model"`
	Messages []OllamaMessage `json:"messages"`
	Stream   *bool           `json:"stream"`
}

// GenerateRequest represents the request for /api/generate endpoint
type GenerateRequest struct {
	Model     string                 `json:"model"`
	Prompt    string                 `json:"prompt"`
	Stream    *bool                  `json:"stream"`
	Options   map[string]interface{} `json:"options"`
	Format    string                 `json:"format"`
	Raw       bool                   `json:"raw"`
	Context   []int                  `json:"context"`
	KeepAlive string                 `json:"keep_alive"`
}

type OpenRouterArchitecture struct {
	InputModalities []string `json:"input_modalities"`
	OutputModalities []string `json:"output_modalities"`
}

type OpenRouterModel struct {
	openai.Model
	Name                string                 `json:"name"`
	Description         string                 `json:"description,omitempty"`
	Architecture        OpenRouterArchitecture `json:"architecture,omitempty"`
	SupportedParameters []string               `json:"supported_parameters,omitempty"`
	ContextLength       int                    `json:"context_length,omitempty"`
}

type OpenRouterModelsList struct {
	Models []OpenRouterModel `json:"data"`
}
