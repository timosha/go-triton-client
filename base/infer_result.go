package base

import (
	"fmt"
	"github.com/Trendyol/go-triton-client/converter"
)

// InferResult is an interface that defines methods for retrieving output data from an inference result.
type InferResult interface {
	GetOutput(name string) (InferOutput, error)
	AsSlice(name string) (interface{}, error)
}

// BaseInferResult provides common fields and methods for InferResult implementations.
type BaseInferResult struct {
	OutputsResponse       InferOutputs
	OutputNameToBufferMap map[string]int
	Buffer                []byte
	DataConverter         converter.DataConverter
}

// GetOutput retrieves the output tensor corresponding to the named output.
func (r *BaseInferResult) GetOutput(name string) (InferOutput, error) {
	if len(r.OutputsResponse.Outputs) == 0 {
		return nil, fmt.Errorf("no outputs found in result")
	}
	for _, output := range r.OutputsResponse.Outputs {
		if output.GetName() == name {
			return output, nil
		}
	}
	return nil, fmt.Errorf("output %s not found", name)
}
