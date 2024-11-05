package http

import (
	"github.com/Trendyol/go-triton-client/base"
	"reflect"
	"testing"
)

func TestNewInferInput(t *testing.T) {
	name := "input0"
	datatype := "FP32"
	shape := []int64{1, 3}
	parameters := map[string]interface{}{"param1": "value1"}
	input := NewInferInput(name, datatype, shape, parameters)
	if input.GetName() != name {
		t.Errorf("Expected Name %s, got %s", name, input.GetName())
	}
	if !reflect.DeepEqual(input.GetShape(), shape) {
		t.Errorf("Expected Shape %v, got %v", shape, input.GetShape())
	}
	if input.GetDatatype() != datatype {
		t.Errorf("Expected Datatype %s, got %s", datatype, input.GetDatatype())
	}
	if !reflect.DeepEqual(input.GetParameters(), parameters) {
		t.Errorf("Expected Parameters %v, got %v", parameters, input.GetParameters())
	}
}

func TestNewInferInput_NilParameters(t *testing.T) {
	name := "input0"
	datatype := "FP32"
	shape := []int64{1, 3}
	input := NewInferInput(name, datatype, shape, nil)
	if input.GetParameters() == nil {
		t.Errorf("Expected Parameters to be initialized, got nil")
	}
}

func TestInferInput_GetTensor(t *testing.T) {
	name := "input0"
	datatype := "FP32"
	shape := []int64{1, 3}
	parameters := map[string]interface{}{"param1": "value1"}
	input := &InferInput{
		BaseInferInput: &base.BaseInferInput{
			Name:       name,
			Datatype:   datatype,
			Shape:      shape,
			Parameters: parameters,
		},
	}
	tensor := input.GetTensor().(map[string]interface{})
	if tensor["name"] != name {
		t.Errorf("Expected tensor name %s, got %s", name, tensor["name"])
	}
	if !reflect.DeepEqual(tensor["shape"], shape) {
		t.Errorf("Expected tensor shape %v, got %v", shape, tensor["shape"])
	}
	if tensor["datatype"] != datatype {
		t.Errorf("Expected tensor datatype %s, got %s", datatype, tensor["datatype"])
	}
	if !reflect.DeepEqual(tensor["parameters"], parameters) {
		t.Errorf("Expected tensor parameters %v, got %v", parameters, tensor["parameters"])
	}
}

func TestInferInput_GetTensor_WithData(t *testing.T) {
	name := "input0"
	datatype := "FP32"
	shape := []int64{1, 3}
	inputData := []interface{}{1.0, 2.0, 3.0}
	input := &InferInput{
		BaseInferInput: &base.BaseInferInput{
			Name:     name,
			Datatype: datatype,
			Shape:    shape,
			Data:     inputData,
			Parameters: map[string]interface{}{
				"param1": "value1",
			},
		},
	}
	tensor := input.GetTensor().(map[string]interface{})
	if tensor["name"] != name {
		t.Errorf("Expected tensor name %s, got %s", name, tensor["name"])
	}
	if !reflect.DeepEqual(tensor["shape"], shape) {
		t.Errorf("Expected tensor shape %v, got %v", shape, tensor["shape"])
	}
	if tensor["datatype"] != datatype {
		t.Errorf("Expected tensor datatype %s, got %s", datatype, tensor["datatype"])
	}
	if !reflect.DeepEqual(tensor["parameters"], input.GetParameters()) {
		t.Errorf("Expected tensor parameters %v, got %v", input.GetParameters(), tensor["parameters"])
	}
	if !reflect.DeepEqual(tensor["data"], inputData) {
		t.Errorf("Expected tensor data %v, got %v", inputData, tensor["data"])
	}
}

func TestInferInput_GetTensor_WithSharedMemory(t *testing.T) {
	name := "input0"
	datatype := "FP32"
	shape := []int64{1, 3}
	input := &InferInput{
		BaseInferInput: &base.BaseInferInput{
			Name:     name,
			Datatype: datatype,
			Shape:    shape,
			Parameters: map[string]interface{}{
				"shared_memory_region": "region0",
			},
			Data: []interface{}{1.0, 2.0, 3.0},
		},
	}
	tensor := input.GetTensor().(map[string]interface{})
	if _, ok := tensor["data"]; ok {
		t.Errorf("Expected tensor data to be omitted when shared_memory_region is set")
	}
}

func TestInferInput_GetTensor_WithRawData(t *testing.T) {
	name := "input0"
	datatype := "FP32"
	shape := []int64{1, 3}
	input := &InferInput{
		BaseInferInput: &base.BaseInferInput{
			Name:     name,
			Datatype: datatype,
			Shape:    shape,
			RawData:  []byte{1, 2, 3, 4},
			Data:     []interface{}{1.0, 2.0, 3.0},
		},
	}
	tensor := input.GetTensor().(map[string]interface{})
	if _, ok := tensor["data"]; ok {
		t.Errorf("Expected tensor data to be omitted when RawData is present")
	}
}

func TestInferInput_GetBinaryData(t *testing.T) {
	rawData := []byte{1, 2, 3, 4}
	input := &InferInput{
		BaseInferInput: &base.BaseInferInput{
			RawData: rawData,
		},
	}
	if !reflect.DeepEqual(input.GetBinaryData(), rawData) {
		t.Errorf("Expected GetBinaryData %v, got %v", rawData, input.GetBinaryData())
	}
}

func TestInferInput_GetBinaryData_Nil(t *testing.T) {
	input := &InferInput{
		BaseInferInput: &base.BaseInferInput{},
	}
	if input.GetBinaryData() != nil {
		t.Errorf("Expected GetBinaryData to be nil, got %v", input.GetBinaryData())
	}
}
