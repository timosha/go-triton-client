package base

// ResponseWrapper is an interface that abstracts the differences between HTTP and gRPC responses.
type ResponseWrapper interface {
	GetHeader(key string) string
	GetBody() ([]byte, error)
	GetRawOutputContents() [][]byte
	GetResponse() any
}
