package grpc

import (
	"github.com/Trendyol/go-triton-client/client/grpc/grpc_generated_v2"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestNewResponseWrapper(t *testing.T) {
	response := &grpc_generated_v2.ModelInferResponse{}
	wrapper := NewResponseWrapper(response)

	_, ok := wrapper.(*ResponseWrapper)
	assert.True(t, ok)
	assert.Equal(t, response, wrapper.(*ResponseWrapper).Response)
}

func TestGetHeader(t *testing.T) {
	response := &ResponseWrapper{
		HeaderLength:    "1234",
		ContentEncoding: "gzip",
	}

	assert.Equal(t, "1234", response.GetHeader("Inference-Header-Content-Length"))
	assert.Equal(t, "gzip", response.GetHeader("Content-Encoding"))
	assert.Equal(t, "", response.GetHeader("Unknown-Header"))
}

func TestGetBodyWithContent(t *testing.T) {
	rawContent := []byte("test body content")
	response := &ResponseWrapper{
		Response: &grpc_generated_v2.ModelInferResponse{
			RawOutputContents: [][]byte{rawContent},
		},
	}

	body, err := response.GetBody()
	assert.NoError(t, err)
	assert.Equal(t, rawContent, body)
}

func TestGetBodyWithoutContent(t *testing.T) {
	response := &ResponseWrapper{
		Response: &grpc_generated_v2.ModelInferResponse{
			RawOutputContents: [][]byte{},
		},
	}

	body, err := response.GetBody()
	assert.Nil(t, body)
	assert.EqualError(t, err, "no body found in gRPC response")
}

func TestGetRawOutputContents(t *testing.T) {
	rawContent := [][]byte{
		[]byte("output 1"),
		[]byte("output 2"),
	}

	response := &ResponseWrapper{
		Response: &grpc_generated_v2.ModelInferResponse{
			RawOutputContents: rawContent,
		},
	}

	assert.Equal(t, rawContent, response.GetRawOutputContents())
}

func TestGetResponse(t *testing.T) {
	mockResponse := &grpc_generated_v2.ModelInferResponse{}
	responseWrapper := &ResponseWrapper{
		Response: mockResponse,
	}

	assert.Equal(t, mockResponse, responseWrapper.GetResponse())
}
