package marshaller

import (
	"encoding/json"
	"github.com/Trendyol/go-triton-client/base"
)

type JSONMarshaller struct{}

// NewJSONMarshaller creates and returns a new instance of JSONMarshaller.
func NewJSONMarshaller() base.Marshaller {
	return &JSONMarshaller{}
}

// Marshal serializes the given value into JSON format using the standard json.Marshal.
func (jm JSONMarshaller) Marshal(v any) ([]byte, error) {
	return json.Marshal(v)
}
