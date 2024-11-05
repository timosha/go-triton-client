package http

import (
	"encoding/json"
	"github.com/Trendyol/go-triton-client/base"
	"github.com/Trendyol/go-triton-client/converter"
)

type InferOutput struct {
	*base.BaseInferOutput
	DataConverter converter.DataConverter
}

// NewInferOutput creates a new instance of InferOutput with the given name and parameters.
func NewInferOutput(name string, parameters map[string]interface{}) base.InferOutput {
	if parameters == nil {
		parameters = make(map[string]interface{})
	}
	return &InferOutput{
		BaseInferOutput: &base.BaseInferOutput{
			Name:       name,
			Parameters: parameters,
		},
		DataConverter: converter.NewDataConverter(),
	}
}

// GetTensor returns a tensor representation of the InferOutput with its name and parameters.
func (output *InferOutput) GetTensor() any {
	tensor := map[string]interface{}{
		"name": output.Name,
	}
	if output.Parameters != nil {
		tensor["parameters"] = output.Parameters
	}
	return tensor
}

// UnmarshalJSON customizes the JSON unmarshaling for InferOutput, handling data conversion
// for different datatypes such as FP32, INT32, BOOL, etc.
func (output *InferOutput) UnmarshalJSON(data []byte) error {
	var tempStruct struct {
		Name           string                 `json:"name"`
		Shape          []int                  `json:"shape"`
		Datatype       string                 `json:"datatype"`
		Parameters     map[string]interface{} `json:"parameters"`
		Data           []interface{}          `json:"data"`
		Classification int                    `json:"classification"`
	}
	if err := json.Unmarshal(data, &tempStruct); err != nil {
		return err
	}
	output.Name = tempStruct.Name
	output.Shape = tempStruct.Shape
	output.Datatype = tempStruct.Datatype
	output.Parameters = tempStruct.Parameters
	switch tempStruct.Datatype {
	case "FP32":
		output.Data = output.DataConverter.ConvertInterfaceSliceToFloat32SliceAsInterface(tempStruct.Data)
	case "FP64":
		output.Data = output.DataConverter.ConvertInterfaceSliceToFloat64SliceAsInterface(tempStruct.Data)
	case "INT32":
		output.Data = output.DataConverter.ConvertInterfaceSliceToInt32SliceAsInterface(tempStruct.Data)
	case "INT64":
		output.Data = output.DataConverter.ConvertInterfaceSliceToInt64SliceAsInterface(tempStruct.Data)
	case "UINT32":
		output.Data = output.DataConverter.ConvertInterfaceSliceToUint32SliceAsInterface(tempStruct.Data)
	case "UINT64":
		output.Data = output.DataConverter.ConvertInterfaceSliceToUint64SliceAsInterface(tempStruct.Data)
	case "BOOL":
		output.Data = output.DataConverter.ConvertInterfaceSliceToBoolSliceAsInterface(tempStruct.Data)
	case "BYTES":
		convertedData, err := output.DataConverter.ConvertInterfaceSliceToBytesSliceAsInterface(tempStruct.Data)
		if err != nil {
			return err
		}
		output.Data = convertedData
	default:
		output.Data = tempStruct.Data
	}
	return nil
}
