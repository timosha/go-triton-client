package base

import (
	"fmt"
	"github.com/Trendyol/go-triton-client/converter"
)

// InferInput is an interface that defines methods for model inference inputs.
type InferInput interface {
	GetName() string
	GetShape() []int64
	GetDatatype() string
	GetParameters() map[string]interface{}
	GetData() []interface{}
	GetRawData() []byte
	GetTensor() any
	GetBinaryData() []byte
	SetDatatype(datatype string)
	SetShape(shape []int64)
	SetData(inputTensor interface{}, binaryData bool) error
}

// BaseInferInput is a base struct that implements common functionality for InferInput.
type BaseInferInput struct {
	Name          string
	Shape         []int64
	Datatype      string
	Parameters    map[string]interface{}
	Data          []interface{}
	RawData       []byte
	DataConverter converter.DataConverter
}

// GetName returns the name of the input tensor.
func (input *BaseInferInput) GetName() string {
	return input.Name
}

// GetShape returns the shape of the input tensor.
func (input *BaseInferInput) GetShape() []int64 {
	return input.Shape
}

// GetDatatype returns the data type of the input tensor.
func (input *BaseInferInput) GetDatatype() string {
	return input.Datatype
}

// GetParameters returns the parameters associated with the input tensor.
func (input *BaseInferInput) GetParameters() map[string]interface{} {
	return input.Parameters
}

// GetData returns the data of the input tensor.
func (input *BaseInferInput) GetData() []interface{} {
	return input.Data
}

// GetRawData returns the raw binary data of the input tensor.
func (input *BaseInferInput) GetRawData() []byte {
	return input.RawData
}

// SetDatatype sets the datatype of the input tensor.
func (input *BaseInferInput) SetDatatype(datatype string) {
	input.Datatype = datatype
}

// SetShape sets the shape of the input tensor.
func (input *BaseInferInput) SetShape(shape []int64) {
	input.Shape = shape
}

// SetData sets the data for the input tensor.
// If binaryData is true, it serializes the inputTensor and stores it as RawData.
// If binaryData is false, it flattens the inputTensor and stores it as Data.
func (input *BaseInferInput) SetData(inputTensor interface{}, binaryData bool) error {
	// Validate the input tensor type matches the expected datatype
	if input.Datatype != getDatatype(inputTensor) {
		return fmt.Errorf("got unexpected datatype %T from input tensor, expected %s", inputTensor, input.Datatype)
	}

	if !binaryData {
		// For non-binary data, remove binary data size parameter and set Data
		delete(input.Parameters, "binary_data_size")
		input.RawData = nil
		input.Data = input.DataConverter.FlattenData(inputTensor)
	} else {
		// For binary data, serialize the tensor and set RawData
		input.Data = nil
		rawData, err := input.DataConverter.SerializeTensor(inputTensor)
		if err != nil {
			return err
		}
		input.RawData = rawData
		input.Parameters["binary_data_size"] = len(rawData)
	}

	return nil
}

// getDatatype determines the data type of the input tensor based on its Go type.
func getDatatype(inputTensor interface{}) string {
	switch inputTensor.(type) {
	case []int:
		return "INT64"
	case []int32:
		return "INT32"
	case []int64:
		return "INT64"
	case []uint16:
		return "UINT16"
	case []uint32:
		return "UINT32"
	case []uint64:
		return "UINT64"
	case []float32:
		return "FP32"
	case []float64:
		return "FP64"
	case []byte:
		return "BYTES"
	case []bool:
		return "BOOL"
	case []string:
		return "BYTES"
	default:
		return "UNKNOWN"
	}
}
