package main

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/sashabaranov/go-openai"
)

type OpenrouterProvider struct {
	client     *openai.Client
	modelNames []string // Shared storage for model names
}

func NewOpenrouterProvider(apiKey string) *OpenrouterProvider {
	config := openai.DefaultConfig(apiKey)
	config.BaseURL = "https://openrouter.ai/api/v1/" // Custom endpoint if needed
	return &OpenrouterProvider{
		client:     openai.NewClientWithConfig(config),
		modelNames: []string{},
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

type ModelDetails struct {
	ParentModel       string   `json:"parent_model"`
	Format            string   `json:"format"`
	Family            string   `json:"family"`
	Families          []string `json:"families"`
	ParameterSize     string   `json:"parameter_size"`
	QuantizationLevel string   `json:"quantization_level"`
}

type Model struct {
	Name       string       `json:"name"`
	Model      string       `json:"model,omitempty"`
	ModifiedAt string       `json:"modified_at,omitempty"`
	Size       int64        `json:"size,omitempty"`
	Digest     string       `json:"digest,omitempty"`
	Details    ModelDetails `json:"details,omitempty"`
}

func (o *OpenrouterProvider) GetModels() ([]Model, error) {
	currentTime := time.Now().Format(time.RFC3339)

	// Fetch models from the OpenAI API
	modelsResponse, err := o.client.ListModels(context.Background())
	if err != nil {
		return nil, err
	}

	// Clear shared model storage
	o.modelNames = []string{}

	var models []Model
	for _, apiModel := range modelsResponse.Models {
		// Split model name
		parts := strings.Split(apiModel.ID, "/")
		name := parts[len(parts)-1]

		// Store name in shared storage
		o.modelNames = append(o.modelNames, apiModel.ID)

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
			"general.architecture":                   "llama",
			"general.file_type":                      2,
			"general.parameter_count":                8030261248,
			"general.quantization_version":           2,
			"context_length":                         200000,
			"parameter_count":                        200_000_000_000,
			"llama.attention.head_count":             32,
			"llama.attention.head_count_kv":          8,
			"llama.attention.layer_norm_rms_epsilon": 0.00001,
			"llama.block_count":                      32,
			"llama.context_length":                   8192,
			"llama.embedding_length":                 4096,
			"llama.feed_forward_length":              14336,
			"llama.rope.dimension_count":             128,
			"llama.rope.freq_base":                   500000,
			"llama.vocab_size":                       128256,
			"tokenizer.ggml.bos_token_id":            128000,
			"tokenizer.ggml.eos_token_id":            128009,
			"tokenizer.ggml.merges":                  []string{}, // populates if `verbose=true`
			"tokenizer.ggml.model":                   "gpt2",
			"tokenizer.ggml.pre":                     "llama-bpe",
			"tokenizer.ggml.token_type":              []string{}, // populates if `verbose=true`
			"tokenizer.ggml.tokens":                  []string{}, // populates if `verbose=true`
		},
		"capabilities": []string{
			"completion",
			"chat",
			"embeddings",
		},
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
