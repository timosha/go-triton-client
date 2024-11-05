package grpc

import (
	"fmt"
	"github.com/Trendyol/go-triton-client/base"
	"github.com/Trendyol/go-triton-client/client/grpc/grpc_generated_v2"
)

// ResponseWrapper wraps a gRPC response to implement UnifiedResponse.
type ResponseWrapper struct {
	Response        *grpc_generated_v2.ModelInferResponse
	HeaderLength    string
	ContentEncoding string
}

func NewResponseWrapper(response *grpc_generated_v2.ModelInferResponse) base.ResponseWrapper {
	return &ResponseWrapper{
		Response: response,
	}
}

// GetHeader retrieves the value of a specified header key from the wrapped gRPC response.
// It checks for "Inference-Header-Content-Length" and "Content-Encoding" headers, returning the appropriate value if found.
// If the key does not match any known headers, it returns an empty string.
func (g *ResponseWrapper) GetHeader(key string) string {
	switch key {
	case "Inference-Header-Content-Length":
		return g.HeaderLength
	case "Content-Encoding":
		return g.ContentEncoding
	default:
		return ""
	}
}

// GetBody retrieves the body content from the gRPC response.
// If there is content available in the RawOutputContents, it returns the first element as a byte slice.
// If no content is found, it returns an error indicating that no body was found.
func (g *ResponseWrapper) GetBody() ([]byte, error) {
	if len(g.Response.RawOutputContents) > 0 {
		return g.Response.RawOutputContents[0], nil
	}
	return nil, fmt.Errorf("no body found in gRPC response")
}

// GetRawOutputContents returns the raw output contents as a slice of byte slices from the gRPC response.
func (g *ResponseWrapper) GetRawOutputContents() [][]byte {
	return g.Response.RawOutputContents
}

// GetResponse retrieves the response.
func (g *ResponseWrapper) GetResponse() any {
	return g.Response
}
