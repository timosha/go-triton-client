package http

import (
	"github.com/Trendyol/go-triton-client/base"
	"io"
	"net/http"
)

// ResponseWrapper wraps an HTTP response to implement the UnifiedResponse interface,
// providing methods to retrieve headers and body content from the HTTP response.
type ResponseWrapper struct {
	Response *http.Response
}

func NewResponseWrapper(response *http.Response) base.ResponseWrapper {
	return &ResponseWrapper{
		Response: response,
	}
}

// GetHeader retrieves the value of a specified header key from the wrapped HTTP response.
// It uses the standard http.Header Get method to return the value associated with the key.
func (h *ResponseWrapper) GetHeader(key string) string {
	return h.Response.Header.Get(key)
}

// GetBody reads and returns the entire body of the HTTP response as a byte slice.
// It returns an error if reading the body fails.
func (h *ResponseWrapper) GetBody() ([]byte, error) {
	return io.ReadAll(h.Response.Body)
}

// GetRawOutputContents reads the entire body of the HTTP response and returns it as a slice of byte slices.
// In this implementation, it returns a slice containing a single byte slice with the full response body.
func (h *ResponseWrapper) GetRawOutputContents() [][]byte {
	body, _ := io.ReadAll(h.Response.Body)
	return [][]byte{body}
}

// GetResponse retrieves the response.
func (g *ResponseWrapper) GetResponse() any {
	return g.Response
}
