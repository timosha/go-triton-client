package grpc

import (
	"fmt"
	"github.com/Trendyol/go-triton-client/base"
	"github.com/Trendyol/go-triton-client/client/grpc/grpc_generated_v2"
)

type InferOutput struct {
	*base.BaseInferOutput
}

// NewInferOutput creates a new instance of InferOutput with the provided name and parameters.
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

// GetTensor generates and returns a ModelInferRequest_InferRequestedOutputTensor based on the output's parameters.
func (output *InferOutput) GetTensor() any {
	requestedOutput := &grpc_generated_v2.ModelInferRequest_InferRequestedOutputTensor{
		Name:       output.Name,
		Parameters: make(map[string]*grpc_generated_v2.InferParameter),
	}
	for key, value := range output.Parameters {
		if key == "binary_data" {
			continue
		}
		switch v := value.(type) {
		case int:
			requestedOutput.Parameters[key] = &grpc_generated_v2.InferParameter{
				ParameterChoice: &grpc_generated_v2.InferParameter_Int64Param{
					Int64Param: int64(v),
				},
			}
		case string:
			requestedOutput.Parameters[key] = &grpc_generated_v2.InferParameter{
				ParameterChoice: &grpc_generated_v2.InferParameter_StringParam{
					StringParam: v,
				},
			}
		default:
			fmt.Printf("unsupported parameter type: %T for key %s", v, key)
		}
	}
	return requestedOutput
}
