package base

import (
	"context"
	"crypto/tls"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/credentials/insecure"
	"time"
)

// GrpcClient interface defines a method to retrieve a gRPC client connection.
type GrpcClient interface {
	GetConnection() *grpc.ClientConn
}

// grpcClient struct holds the gRPC connection instance.
type grpcClient struct {
	grpcConnection *grpc.ClientConn
}

// NewGrpcClient creates a new gRPC client connection with the given parameters.
func NewGrpcClient(url string, connectionTimeout float64, networkTimeout float64, ssl bool, insecureConnection bool) (GrpcClient, error) {
	// Prepare dial options
	var opts []grpc.DialOption

	// Add SSL/TLS credentials
	if ssl {
		var creds credentials.TransportCredentials
		if insecureConnection {
			creds = credentials.NewTLS(&tls.Config{InsecureSkipVerify: true})
		} else {
			creds = credentials.NewTLS(&tls.Config{})
		}
		opts = append(opts, grpc.WithTransportCredentials(creds))
	} else {
		opts = append(opts, grpc.WithTransportCredentials(insecure.NewCredentials()))
	}

	// Add connection timeout
	if connectionTimeout > 0 {
		opts = append(opts, grpc.WithConnectParams(grpc.ConnectParams{
			MinConnectTimeout: time.Duration(connectionTimeout * float64(time.Second)),
		}))
	}

	// Prepare context with network timeout
	ctx := context.Background()
	if networkTimeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, time.Duration(networkTimeout*float64(time.Second)))
		defer cancel()
	}

	// Create gRPC client
	grpcConnection, err := grpc.NewClient(url, opts...)
	if err != nil {
		return nil, err
	}

	return &grpcClient{grpcConnection: grpcConnection}, nil
}

// GetConnection returns the gRPC connection instance stored in the grpcClient struct.
func (g *grpcClient) GetConnection() *grpc.ClientConn {
	return g.grpcConnection
}
