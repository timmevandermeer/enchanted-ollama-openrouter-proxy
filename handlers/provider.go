package handlers

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/sashabaranov/go-openai"
)

type OpenrouterProvider struct {
	client           *openai.Client
	customHttpClient *HTTPClient
	modelNames       []string // Shared storage for model names
	models           []OpenRouterModel
}

func NewOpenrouterProvider(apiKey string) *OpenrouterProvider {
	config := openai.DefaultConfig(apiKey)
	config.BaseURL = "https://openrouter.ai/api/v1/" // Custom endpoint if needed

	baseURL := "https://openrouter.ai/api/v1/"
	httpClient := NewHTTPClient(baseURL, apiKey)

	return &OpenrouterProvider{
		client:           openai.NewClientWithConfig(config),
		customHttpClient: httpClient,
		modelNames:       []string{},
		models:           []OpenRouterModel{},
	}
}

func (o *OpenrouterProvider) Chat(messages []openai.ChatCompletionMessage, modelName string) (openai.ChatCompletionResponse, error) {
	// Create a chat completion request
	req := openai.ChatCompletionRequest{
		Model:    modelName,
		Messages: messages,
		Stream:   false,
	}

	// Call the OpenAI API to get a complete response
	resp, err := o.client.CreateChatCompletion(context.Background(), req)
	if err != nil {
		return openai.ChatCompletionResponse{}, err
	}

	// Return the complete response
	return resp, nil
}

func (o *OpenrouterProvider) ChatStream(messages []openai.ChatCompletionMessage, modelName string) (*openai.ChatCompletionStream, error) {
	// Create a chat completion request
	req := openai.ChatCompletionRequest{
		Model:    modelName,
		Messages: messages,
		Stream:   true,
	}

	// Call the OpenAI API to get a streaming response
	stream, err := o.client.CreateChatCompletionStream(context.Background(), req)
	if err != nil {
		return nil, err
	}

	// Return the stream for further processing
	return stream, nil
}

func (o *OpenrouterProvider) GetModels() ([]Model, error) {
	currentTime := time.Now().Format(time.RFC3339)

	// Fetch models from the OpenAI API
	customResponse, err := o.customHttpClient.Get("models")
	var modelsResponse OpenRouterModelsList
	o.customHttpClient.DecodeJSON(customResponse, &modelsResponse)

	if err != nil {
		return nil, err
	}

	// Clear shared model storage
	o.modelNames = []string{}
	o.models = []OpenRouterModel{}

	var models []Model
	for _, apiModel := range modelsResponse.Models {
		// Split model name
		parts := strings.Split(apiModel.ID, "/")
		name := parts[len(parts)-1]

		// Store name in shared storage
		o.modelNames = append(o.modelNames, apiModel.ID)
		o.models = append(o.models, apiModel)

		// Create model struct
		model := Model{
			Name:       name,
			Model:      name,
			ModifiedAt: currentTime,
			Size:       0, // Stubbed size
			Digest:     name,
			Details: ModelDetails{
				ParentModel:       "",
				Format:            "gguf",
				Family:            "claude",
				Families:          []string{"claude"},
				ParameterSize:     "175B",
				QuantizationLevel: "Q4_K_M",
			},
		}
		models = append(models, model)
	}

	return models, nil
}

func (o *OpenrouterProvider) GetModelDetails(modelName string) (map[string]interface{}, error) {
	// Stub response; replace with actual model details if available
	currentTime := time.Now().Format(time.RFC3339)

	// Find model info in shared storage
	modelInfo := OpenRouterModel{}
	fullModelName, _ := o.GetFullModelName(modelName)

	for _, model := range o.models {
		if model.ID == fullModelName {
			modelInfo = model
			break
		}
	}

	contextLength := 128000 // Default context length if not found
	if modelInfo.ContextLength != 0 {
		contextLength = modelInfo.ContextLength
	}

	capatbilities := []string{
		"completion",
		"chat",
	}

	for _, modality := range modelInfo.Architecture.InputModalities {
		if modality == "image" {
			capatbilities = append(capatbilities, "vision")
		}
	}

	return map[string]interface{}{
		"license":    "STUB License",
		"system":     "STUB SYSTEM",
		"modifiedAt": currentTime,
		"details": map[string]interface{}{
			"parent_model":       "",
			"format":             "gguf",
			"parameter_size":     "200B",
			"quantization_level": "Q4_K_M",
			"family":             "llama",
		},
		"model_info": map[string]interface{}{
			"llama.context_length": contextLength,
		},
		"capabilities": capatbilities,
	}, nil
}

func (o *OpenrouterProvider) GetFullModelName(alias string) (string, error) {
	// If modelNames is empty or not populated yet, try to get models first
	if len(o.modelNames) == 0 {
		_, err := o.GetModels()
		if err != nil {
			return "", fmt.Errorf("failed to get models: %w", err)
		}
	}

	// First try exact match
	for _, fullName := range o.modelNames {
		if fullName == alias {
			return fullName, nil
		}
	}

	// Then try suffix match
	for _, fullName := range o.modelNames {
		if strings.HasSuffix(fullName, alias) {
			return fullName, nil
		}
	}

	// If no match found, just use the alias as is
	// This allows direct use of model names that might not be in the list
	return alias, nil
}

func (o *OpenrouterProvider) Generate(prompt, modelName string, options map[string]interface{}) (openai.CompletionResponse, error) {
	// Create a completion request (not chat completion)
	req := openai.CompletionRequest{
		Model:  modelName,
		Prompt: prompt,
		Stream: false,
	}

	// Apply options if provided
	if options != nil {
		if temp, ok := options["temperature"].(float64); ok {
			req.Temperature = float32(temp)
		}
		if maxTokens, ok := options["num_predict"].(int); ok {
			req.MaxTokens = maxTokens
		}
		if topP, ok := options["top_p"].(float64); ok {
			req.TopP = float32(topP)
		}
		if seed, ok := options["seed"].(int); ok {
			req.Seed = &seed
		}
		if stop, ok := options["stop"].([]string); ok {
			req.Stop = stop
		}
	}

	// Call the OpenAI API to get a complete response
	resp, err := o.client.CreateCompletion(context.Background(), req)
	if err != nil {
		return openai.CompletionResponse{}, err
	}

	return resp, nil
}

func (o *OpenrouterProvider) GenerateStream(prompt, modelName string, options map[string]interface{}) (*openai.CompletionStream, error) {
	// Create a completion request (not chat completion)
	req := openai.CompletionRequest{
		Model:  modelName,
		Prompt: prompt,
		Stream: true,
	}

	// Apply options if provided
	if options != nil {
		if temp, ok := options["temperature"].(float64); ok {
			req.Temperature = float32(temp)
		}
		if maxTokens, ok := options["num_predict"].(int); ok {
			req.MaxTokens = maxTokens
		}
		if topP, ok := options["top_p"].(float64); ok {
			req.TopP = float32(topP)
		}
		if seed, ok := options["seed"].(int); ok {
			req.Seed = &seed
		}
		if stop, ok := options["stop"].([]string); ok {
			req.Stop = stop
		}
	}

	// Call the OpenAI API to get a streaming response
	stream, err := o.client.CreateCompletionStream(context.Background(), req)
	if err != nil {
		return nil, err
	}

	return stream, nil
}
