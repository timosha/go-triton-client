package grpc

import (
	"github.com/Trendyol/go-triton-client/base"
	"github.com/Trendyol/go-triton-client/client/grpc/grpc_generated_v2"
	"github.com/Trendyol/go-triton-client/converter"
	"log"
)

// InferInput is the gRPC implementation of the base.InferInput interface.
type InferInput struct {
	*base.BaseInferInput
}

// NewInferInput creates a new gRPC InferInput instance with the given parameters.
func NewInferInput(name string, datatype string, shape []int64, parameters map[string]interface{}) base.InferInput {
	if parameters == nil {
		parameters = make(map[string]interface{})
	}
	return &InferInput{
		BaseInferInput: &base.BaseInferInput{
			Name:          name,
			Shape:         shape,
			Datatype:      datatype,
			Parameters:    parameters,
			DataConverter: converter.NewDataConverter(),
		},
	}
}

// GetTensor constructs the tensor representation suitable for the gRPC inference request.
func (input *InferInput) GetTensor() any {
	inputTensor := &grpc_generated_v2.ModelInferRequest_InferInputTensor{
		Name:     input.Name,
		Datatype: input.Datatype,
		Shape:    input.Shape,
	}

	if len(input.RawData) > 0 {
		return inputTensor
	}

	contents := &grpc_generated_v2.InferTensorContents{}

	switch input.Datatype {
	case "INT8", "INT16", "INT32":
		contents.IntContents = input.DataConverter.ConvertByteSliceToInt32Slice(input.GetBinaryData())
	case "INT64":
		contents.Int64Contents = input.DataConverter.ConvertByteSliceToInt64Slice(input.GetBinaryData())
	case "FP32":
		contents.Fp32Contents = input.DataConverter.ConvertByteSliceToFloat32Slice(input.GetBinaryData())
	case "FP64":
		contents.Fp64Contents = input.DataConverter.ConvertByteSliceToFloat64Slice(input.GetBinaryData())
	case "BYTES":
		contents.BytesContents = [][]byte{input.GetBinaryData()}
	default:
		contents = nil
		log.Printf("unsupported datatype: %s", input.Datatype)
	}
	inputTensor.Contents = contents

	return inputTensor
}

// GetBinaryData returns the raw binary data of the input tensor.
func (input *InferInput) GetBinaryData() []byte {
	return input.RawData
}
