package base

import (
	"github.com/stretchr/testify/assert"
	"google.golang.org/grpc"
	"testing"
)

func TestNewGrpcClient(t *testing.T) {
	tests := []struct {
		name               string
		url                string
		connectionTimeout  float64
		networkTimeout     float64
		ssl                bool
		insecureConnection bool
		expectError        bool
	}{
		{
			name:               "SSL with insecureConnection",
			url:                "test:50051",
			connectionTimeout:  10.0,
			networkTimeout:     20.0,
			ssl:                true,
			insecureConnection: true,
			expectError:        false,
		},
		{
			name:               "SSL without insecureConnection",
			url:                "test:50051",
			connectionTimeout:  10.0,
			networkTimeout:     20.0,
			ssl:                true,
			insecureConnection: false,
			expectError:        false,
		},
		{
			name:               "No SSL",
			url:                "test:50051",
			connectionTimeout:  10.0,
			networkTimeout:     20.0,
			ssl:                false,
			insecureConnection: false,
			expectError:        false,
		},
		{
			name:               "Invalid URL",
			url:                "invalid_url\n",
			connectionTimeout:  10.0,
			networkTimeout:     20.0,
			ssl:                false,
			insecureConnection: false,
			expectError:        true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client, err := NewGrpcClient(tt.url, tt.connectionTimeout, tt.networkTimeout, tt.ssl, tt.insecureConnection)
			if tt.expectError {
				assert.Error(t, err)
				assert.Nil(t, client)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, client)
			}
		})
	}
}

func TestGrpcClient_GetConnection(t *testing.T) {
	conn := &grpc.ClientConn{}
	client := &grpcClient{grpcConnection: conn}
	result := client.GetConnection()
	assert.Equal(t, conn, result)
}
