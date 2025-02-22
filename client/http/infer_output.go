package http

import (
	"github.com/Trendyol/go-triton-client/base"
)

type InferOutput struct {
	*base.BaseInferOutput
}

// NewInferOutput creates a new instance of InferOutput with the given name and parameters.
func NewInferOutput(name string, parameters map[string]any) base.InferOutput {
	if parameters == nil {
		parameters = make(map[string]any)
	}
	return &InferOutput{
		BaseInferOutput: &base.BaseInferOutput{
			Name:       name,
			Parameters: parameters,
		},
	}
}

// GetTensor returns a tensor representation of the InferOutput with its name and parameters.
func (output *InferOutput) GetTensor() any {
	tensor := map[string]any{
		"name": output.Name,
	}
	if output.Parameters != nil {
		tensor["parameters"] = output.Parameters
	}
	return tensor
}
