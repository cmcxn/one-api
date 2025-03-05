package controller

import (
	"bytes"
	"encoding/json"
	"fmt"
	"github.com/gin-gonic/gin"
	"github.com/songquanpeng/one-api/common/logger"
	"github.com/songquanpeng/one-api/relay"
	"github.com/songquanpeng/one-api/relay/adaptor"
	"github.com/songquanpeng/one-api/relay/adaptor/openai"
	"github.com/songquanpeng/one-api/relay/apitype"
	"github.com/songquanpeng/one-api/relay/billing"
	billingratio "github.com/songquanpeng/one-api/relay/billing/ratio"
	"github.com/songquanpeng/one-api/relay/channeltype"
	"github.com/songquanpeng/one-api/relay/meta"
	"github.com/songquanpeng/one-api/relay/model"
	"golang.org/x/net/context"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"
)

// responseBodyLogWriter is a writer that captures the response body
type responseBodyLogWriter struct {
	gin.ResponseWriter
	body      *bytes.Buffer
	isStream  bool
	streamMux sync.Mutex
}

func (w *responseBodyLogWriter) Write(b []byte) (int, error) {
	if w.isStream {
		w.streamMux.Lock()
		defer w.streamMux.Unlock()
	}
	w.body.Write(b)
	return w.ResponseWriter.Write(b)
}

func (w *responseBodyLogWriter) WriteString(s string) (int, error) {
	if w.isStream {
		w.streamMux.Lock()
		defer w.streamMux.Unlock()
	}
	w.body.WriteString(s)
	return w.ResponseWriter.WriteString(s)
}

func RelayTextHelper(c *gin.Context) *model.ErrorWithStatusCode {
	ctx := c.Request.Context()
	meta := meta.GetByContext(c)
	// get & validate textRequest
	textRequest, err := getAndValidateTextRequest(c, meta.Mode)
	if err != nil {
		logger.Errorf(ctx, "getAndValidateTextRequest failed: %s", err.Error())
		return openai.ErrorWrapper(err, "invalid_text_request", http.StatusBadRequest)
	}
	meta.IsStream = textRequest.Stream

	// Wrap the response writer to capture the response
	responseBodyBuffer := &bytes.Buffer{}
	writer := &responseBodyLogWriter{
		ResponseWriter: c.Writer,
		body:           responseBodyBuffer,
		isStream:       meta.IsStream,
	}
	c.Writer = writer

	// map model name
	var isModelMapped bool
	meta.OriginModelName = textRequest.Model
	textRequest.Model, isModelMapped = getMappedModelName(textRequest.Model, meta.ModelMapping)
	meta.ActualModelName = textRequest.Model
	// get model ratio & group ratio
	modelRatio := billingratio.GetModelRatio(textRequest.Model)
	groupRatio := billingratio.GetGroupRatio(meta.Group)
	ratio := modelRatio * groupRatio
	// pre-consume quota
	promptTokens := getPromptTokens(textRequest, meta.Mode)
	meta.PromptTokens = promptTokens
	preConsumedQuota, bizErr := preConsumeQuota(ctx, textRequest, promptTokens, ratio, meta)
	if bizErr != nil {
		logger.Warnf(ctx, "preConsumeQuota failed: %+v", *bizErr)
		return bizErr
	}

	adaptor := relay.GetAdaptor(meta.APIType)
	if adaptor == nil {
		return openai.ErrorWrapper(fmt.Errorf("invalid api type: %d", meta.APIType), "invalid_api_type", http.StatusBadRequest)
	}
	adaptor.Init(meta)

	// get request body
	requestBody, bodyContent, err := getRequestBody(c, meta, textRequest, adaptor, isModelMapped)
	if err != nil {
		return openai.ErrorWrapper(err, "convert_request_failed", http.StatusInternalServerError)
	}
	// Log the final request body
	currentTime := time.Now().Format("2006-01-02 15:04:05")
	logger.Infof(ctx, "[%s] Final request body: <requestBody> %s</requestBody>", currentTime, bodyContent)

	// do request
	resp, err := adaptor.DoRequest(c, meta, requestBody)
	if err != nil {
		logger.Errorf(ctx, "DoRequest failed: %s", err.Error())
		return openai.ErrorWrapper(err, "do_request_failed", http.StatusInternalServerError)
	}
	if isErrorHappened(meta, resp) {
		billing.ReturnPreConsumedQuota(ctx, preConsumedQuota, meta.TokenId)
		return RelayErrorHandler(resp)
	}

	// do response
	usage, respErr := adaptor.DoResponse(c, resp, meta)
	if respErr != nil {
		logger.Errorf(ctx, "respErr is not nil: %+v", respErr)
		billing.ReturnPreConsumedQuota(ctx, preConsumedQuota, meta.TokenId)
		return respErr
	}

	// Log the response body
	currentTime = time.Now().Format("2006-01-02 15:04:05")
	logResponseBody(ctx, responseBodyBuffer.String(), meta.IsStream, currentTime)

	// post-consume quota
	go postConsumeQuota(ctx, usage, meta, textRequest, ratio, preConsumedQuota, modelRatio, groupRatio)
	return nil
}

func getRequestBody(c *gin.Context, meta *meta.Meta, textRequest *model.GeneralOpenAIRequest, adaptor adaptor.Adaptor, isModelMapped bool) (io.Reader, string, error) {
	ctx := c.Request.Context()
	var requestBody io.Reader
	var bodyContent string

	if meta.APIType == apitype.OpenAI {
		// no need to convert request for openai
		shouldResetRequestBody := isModelMapped || meta.ChannelType == channeltype.Baichuan // frequency_penalty 0 is not acceptable for baichuan
		if shouldResetRequestBody {
			jsonStr, err := json.Marshal(textRequest)
			if err != nil {
				return nil, "", err
			}
			bodyContent = string(jsonStr)
			requestBody = bytes.NewBuffer(jsonStr)
		} else {
			// Read and store the body for logging
			bodyBytes, err := io.ReadAll(c.Request.Body)
			if err != nil {
				return nil, "", err
			}
			bodyContent = string(bodyBytes)
			// Restore the body for further processing
			c.Request.Body = io.NopCloser(bytes.NewBuffer(bodyBytes))
			requestBody = bytes.NewBuffer(bodyBytes)
		}
	} else {
		convertedRequest, err := adaptor.ConvertRequest(c, meta.Mode, textRequest)
		if err != nil {
			logger.Debugf(ctx, "converted request failed: %s\n", err.Error())
			return nil, "", err
		}
		jsonData, err := json.Marshal(convertedRequest)
		if err != nil {
			logger.Debugf(ctx, "converted request json_marshal_failed: %s\n", err.Error())
			return nil, "", err
		}
		logger.Debugf(ctx, "converted request: \n%s", string(jsonData))
		bodyContent = string(jsonData)
		requestBody = bytes.NewBuffer(jsonData)
	}

	return requestBody, bodyContent, nil
}

// logResponseBody handles logging the response body with appropriate processing
func logResponseBody(ctx context.Context, responseBody string, isStream bool, timestamp string) {
	if responseBody == "" {
		logger.Infof(ctx, "[%s] Empty response body", timestamp)
		return
	}

	if isStream {
		// For stream responses, extract content only
		content := extractContentFromStream(responseBody)
		logger.Infof(ctx, "[%s] Extracted content:<responseBody> %s</responseBody>", timestamp, content)
	} else {
		// For non-stream responses, extract content
		content := extractContentFromResponse(responseBody)
		logger.Infof(ctx, "[%s] Extracted content:<responseBody> %s</responseBody>", timestamp, content)
	}
}

// extractContentFromResponse extracts only the content field from a non-streaming response
func extractContentFromResponse(responseBody string) string {
	var jsonData map[string]interface{}
	if err := json.Unmarshal([]byte(responseBody), &jsonData); err != nil {
		return "Failed to parse response JSON"
	}

	choices, ok := jsonData["choices"].([]interface{})
	if !ok || len(choices) == 0 {
		return "No content found in response"
	}

	choice, ok := choices[0].(map[string]interface{})
	if !ok {
		return "Invalid choice format in response"
	}

	message, ok := choice["message"].(map[string]interface{})
	if !ok {
		return "Invalid message format in response"
	}

	content, ok := message["content"].(string)
	if !ok {
		return "No content field found in message"
	}

	return content
}

// extractContentFromStream extracts and combines content from a streaming response
func extractContentFromStream(content string) string {
	// Split by "data: " to get individual chunks
	chunks := strings.Split(content, "data: ")

	var combinedContent strings.Builder

	for _, chunk := range chunks {
		chunk = strings.TrimSpace(chunk)
		if chunk == "" || chunk == "[DONE]" {
			continue
		}

		// Parse JSON content
		var jsonData map[string]interface{}
		err := json.Unmarshal([]byte(chunk), &jsonData)
		if err != nil {
			continue // Skip if not valid JSON
		}

		// Extract content from choices
		choices, ok := jsonData["choices"].([]interface{})
		if !ok || len(choices) == 0 {
			continue
		}

		for _, choice := range choices {
			choiceMap, ok := choice.(map[string]interface{})
			if !ok {
				continue
			}

			delta, ok := choiceMap["delta"].(map[string]interface{})
			if !ok {
				continue
			}

			// Extract content piece
			contentPiece, ok := delta["content"].(string)
			if ok {
				combinedContent.WriteString(contentPiece)
			}
		}
	}

	return combinedContent.String()
}
