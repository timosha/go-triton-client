package grpc

import (
	"fmt"
	"github.com/Trendyol/go-triton-client/base"
	"github.com/Trendyol/go-triton-client/client/grpc/grpc_generated_v2"
	"github.com/Trendyol/go-triton-client/converter"
)

// InferResult represents the result of an inference operation using gRPC.
type InferResult struct {
	*base.BaseInferResult
}

// NewInferResult creates a new gRPC InferResult instance.
func NewInferResult(responseWrapper base.ResponseWrapper, dataConverter converter.DataConverter, verbose bool) (base.InferResult, error) {
	result := base.InferOutputs{}
	buffer := []byte{}
	outputNameToBufferMap := make(map[string]int)

	response, ok := responseWrapper.GetResponse().(*grpc_generated_v2.ModelInferResponse)
	if !ok {
		return nil, fmt.Errorf("invalid response type")
	}
	result.ModelName = response.ModelName
	result.ModelVersion = response.ModelVersion

	bufferIndex := 0
	outputs := make([]*base.BaseInferOutput, len(response.Outputs))
	for i, output := range response.Outputs {
		dataBuffer := response.RawOutputContents[i]

		modelOutput := &base.BaseInferOutput{
			Name:     output.Name,
			Datatype: output.Datatype,
			Shape:    convertInt64ToInt(output.Shape),
		}

		outputNameToBufferMap[output.GetName()] = bufferIndex
		bufferIndex += len(dataBuffer)
		buffer = append(buffer, dataBuffer...)

		outputs[i] = modelOutput
	}

	result.Outputs = outputs

	if verbose {
		fmt.Println("GRPC Response:", response)
		fmt.Println("Buffer:", buffer)
	}

	return &InferResult{
		BaseInferResult: &base.BaseInferResult{
			OutputsResponse:       result,
			OutputNameToBufferMap: outputNameToBufferMap,
			Buffer:                buffer,
			DataConverter:         dataConverter,
		},
	}, nil
}

// AsSlice retrieves the output tensor by name, extracts its corresponding data from the buffer,
// deserializes it using the DataConverter, and reshapes it into the appropriate slice type.
// It returns the reshaped slice or an error if the process fails.
func (r *InferResult) AsSlice(name string) (interface{}, error) {
	output, err := r.GetOutput(name)
	if err != nil {
		return nil, err
	}

	startIndex := r.OutputNameToBufferMap[name]
	endIndex := startIndex + len(r.Buffer[startIndex:])

	dataBuffer := r.Buffer[startIndex:endIndex]

	slice, err := r.DataConverter.DeserializeTensor(output.GetDatatype(), dataBuffer)
	if err != nil {
		return nil, err
	}

	return r.DataConverter.ReshapeArray(slice, output.GetShape())
}

func convertInt64ToInt(int64Slice []int64) []int {
	intSlice := make([]int, len(int64Slice))
	for i, v := range int64Slice {
		intSlice[i] = int(v)
	}
	return intSlice
}
