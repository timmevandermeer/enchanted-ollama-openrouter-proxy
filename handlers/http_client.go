package handlers

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// HTTPClient wraps the standard http.Client with OpenRouter-specific configuration
type HTTPClient struct {
	client  *http.Client
	baseURL string
	apiKey  string
}

// NewHTTPClient creates a new HTTP client configured for OpenRouter API
func NewHTTPClient(baseURL, apiKey string) *HTTPClient {
	return &HTTPClient{
		client: &http.Client{
			Timeout: 30 * time.Second,
		},
		baseURL: baseURL,
		apiKey:  apiKey,
	}
}

// Request makes a generic HTTP request to the OpenRouter API
func (h *HTTPClient) Request(method, endpoint string, body interface{}) (*http.Response, error) {
	url := h.baseURL + endpoint

	var reqBody io.Reader
	if body != nil {
		jsonBody, err := json.Marshal(body)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal request body: %w", err)
		}
		reqBody = bytes.NewBuffer(jsonBody)
	}

	req, err := http.NewRequest(method, url, reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// Set headers
	req.Header.Set("Authorization", "Bearer "+h.apiKey)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", "enchanted-ollama-openrouter-proxy/1.0")

	resp, err := h.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to make request: %w", err)
	}

	return resp, nil
}

// Get makes a GET request to the specified endpoint
func (h *HTTPClient) Get(endpoint string) (*http.Response, error) {
	return h.Request("GET", endpoint, nil)
}

// Post makes a POST request to the specified endpoint with the given body
func (h *HTTPClient) Post(endpoint string, body interface{}) (*http.Response, error) {
	return h.Request("POST", endpoint, body)
}

// Put makes a PUT request to the specified endpoint with the given body
func (h *HTTPClient) Put(endpoint string, body interface{}) (*http.Response, error) {
	return h.Request("PUT", endpoint, body)
}

// Delete makes a DELETE request to the specified endpoint
func (h *HTTPClient) Delete(endpoint string) (*http.Response, error) {
	return h.Request("DELETE", endpoint, nil)
}

// DecodeJSON decodes JSON response body into the provided interface
func (h *HTTPClient) DecodeJSON(resp *http.Response, v interface{}) error {
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(body))
	}

	return json.NewDecoder(resp.Body).Decode(v)
}

// ReadBody reads and returns the response body as bytes
func (h *HTTPClient) ReadBody(resp *http.Response) ([]byte, error) {
	defer resp.Body.Close()
	return io.ReadAll(resp.Body)
}
