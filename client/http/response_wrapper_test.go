package http

import (
	"github.com/stretchr/testify/assert"
	"io"
	"net/http"
	"strings"
	"testing"
)

func TestNewResponseWrapper(t *testing.T) {
	mockResponse := &http.Response{}
	wrapper := NewResponseWrapper(mockResponse)

	_, ok := wrapper.(*ResponseWrapper)
	assert.True(t, ok)
	assert.Equal(t, mockResponse, wrapper.(*ResponseWrapper).Response)
}

func TestGetHeader(t *testing.T) {
	mockResponse := &http.Response{
		Header: http.Header{
			"Content-Type":  []string{"application/json"},
			"Custom-Header": []string{"custom-value"},
		},
	}

	wrapper := NewResponseWrapper(mockResponse)

	assert.Equal(t, "application/json", wrapper.GetHeader("Content-Type"))
	assert.Equal(t, "custom-value", wrapper.GetHeader("Custom-Header"))
	assert.Equal(t, "", wrapper.GetHeader("Non-Existing-Header"))
}

func TestGetBodySuccess(t *testing.T) {
	bodyContent := "test body content"
	mockResponse := &http.Response{
		Body: io.NopCloser(strings.NewReader(bodyContent)),
	}

	wrapper := NewResponseWrapper(mockResponse)
	body, err := wrapper.GetBody()

	assert.NoError(t, err)
	assert.Equal(t, []byte(bodyContent), body)
}

func TestGetBodyError(t *testing.T) {
	pr, pw := io.Pipe()
	pw.CloseWithError(io.ErrUnexpectedEOF)

	mockResponse := &http.Response{
		Body: io.NopCloser(pr),
	}

	wrapper := NewResponseWrapper(mockResponse)
	body, err := wrapper.GetBody()

	assert.Equal(t, body, []byte{})
	assert.Error(t, err)
}

func TestGetRawOutputContents(t *testing.T) {
	bodyContent := "raw output content"
	mockResponse := &http.Response{
		Body: io.NopCloser(strings.NewReader(bodyContent)),
	}

	wrapper := NewResponseWrapper(mockResponse)
	rawContents := wrapper.GetRawOutputContents()

	assert.Equal(t, 1, len(rawContents))
	assert.Equal(t, []byte(bodyContent), rawContents[0])
}

func TestGetResponse(t *testing.T) {
	mockResponse := &http.Response{}
	wrapper := NewResponseWrapper(mockResponse)

	assert.Equal(t, mockResponse, wrapper.GetResponse())
}
