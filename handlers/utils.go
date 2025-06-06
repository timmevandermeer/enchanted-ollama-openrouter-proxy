package handlers

import (
	"strings"

	openai "github.com/sashabaranov/go-openai"
)

// convertOllamaMessagesToOpenAI converts Ollama-format messages to OpenAI format
func ConvertOllamaMessagesToOpenAI(ollamaMessages []OllamaMessage) []openai.ChatCompletionMessage {
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
