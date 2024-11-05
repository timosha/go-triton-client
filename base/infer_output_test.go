package base

import (
	"reflect"
	"testing"
)

func TestBaseInferOutput_GetName(t *testing.T) {
	output := &BaseInferOutput{Name: "output0"}
	if output.GetName() != "output0" {
		t.Errorf("Expected Name 'output0', got %s", output.GetName())
	}
}

func TestBaseInferOutput_GetShape(t *testing.T) {
	shape := []int{1, 3}
	output := &BaseInferOutput{Shape: shape}
	if !reflect.DeepEqual(output.GetShape(), shape) {
		t.Errorf("Expected Shape %v, got %v", shape, output.GetShape())
	}
}

func TestBaseInferOutput_GetDatatype(t *testing.T) {
	output := &BaseInferOutput{Datatype: "FP32"}
	if output.GetDatatype() != "FP32" {
		t.Errorf("Expected Datatype 'FP32', got %s", output.GetDatatype())
	}
}

func TestBaseInferOutput_GetParameters(t *testing.T) {
	parameters := map[string]interface{}{"param1": "value1"}
	output := &BaseInferOutput{Parameters: parameters}
	if !reflect.DeepEqual(output.GetParameters(), parameters) {
		t.Errorf("Expected Parameters %v, got %v", parameters, output.GetParameters())
	}
}

func TestBaseInferOutput_GetData(t *testing.T) {
	data := []interface{}{1, 2, 3}
	output := &BaseInferOutput{Data: data}
	if !reflect.DeepEqual(output.GetData(), data) {
		t.Errorf("Expected Data %v, got %v", data, output.GetData())
	}
}

func TestBaseInferOutput_GetTensor(t *testing.T) {
	data := []interface{}{1, 2, 3}
	output := &BaseInferOutput{Data: data}
	result := output.GetTensor()
	if parsedResult, ok := result.(error); !ok {
		t.Errorf("expecting an error got %v", parsedResult)
	}
}
