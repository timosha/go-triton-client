package grpc

import (
	"fmt"
	"github.com/Trendyol/go-triton-client/base"
	"github.com/Trendyol/go-triton-client/client/grpc/grpc_generated_v2"
	"io"
	"os"
	"reflect"
	"strings"
	"testing"
)

func TestNewInferOutput(t *testing.T) {
	name := "output0"
	parameters := map[string]interface{}{"param1": "value1"}
	output := NewInferOutput(name, parameters)
	if output.GetName() != name {
		t.Errorf("Expected Name %s, got %s", name, output.GetName())
	}
	if !reflect.DeepEqual(output.GetParameters(), parameters) {
		t.Errorf("Expected Parameters %v, got %v", parameters, output.GetParameters())
	}
}

func TestNewInferOutput_NilParameters(t *testing.T) {
	name := "output0"
	output := NewInferOutput(name, nil)
	if output.GetParameters() == nil {
		t.Errorf("Expected Parameters to be initialized, got nil")
	}
}

func TestInferOutput_GetTensor(t *testing.T) {
	name := "output0"
	parameters := map[string]interface{}{
		"param1":      "value1",
		"param2":      42,
		"binary_data": true,
	}
	output := &InferOutput{
		BaseInferOutput: &base.BaseInferOutput{
			Name:       name,
			Parameters: parameters,
		},
	}
	tensor := output.GetTensor().(*grpc_generated_v2.ModelInferRequest_InferRequestedOutputTensor)
	if tensor.Name != name {
		t.Errorf("Expected Name %s, got %s", name, tensor.Name)
	}
	expectedParameters := map[string]*grpc_generated_v2.InferParameter{
		"param1": {
			ParameterChoice: &grpc_generated_v2.InferParameter_StringParam{
				StringParam: "value1",
			},
		},
		"param2": {
			ParameterChoice: &grpc_generated_v2.InferParameter_Int64Param{
				Int64Param: 42,
			},
		},
	}
	if !reflect.DeepEqual(tensor.Parameters, expectedParameters) {
		t.Errorf("Expected Parameters %v, got %v", expectedParameters, tensor.Parameters)
	}
}

func TestInferOutput_GetTensor_UnsupportedParameterType(t *testing.T) {
	name := "output0"
	parameters := map[string]interface{}{
		"param1": []int{1, 2, 3},
	}
	output := &InferOutput{
		BaseInferOutput: &base.BaseInferOutput{
			Name:       name,
			Parameters: parameters,
		},
	}
	oldStdout := os.Stdout
	r, w, _ := os.Pipe()
	os.Stdout = w
	tensor := output.GetTensor().(*grpc_generated_v2.ModelInferRequest_InferRequestedOutputTensor)
	if len(tensor.Parameters) != 0 {
		t.Errorf("Expected no parameters due to unsupported type, got %v", tensor.Parameters)
	}
	w.Close()
	out, _ := io.ReadAll(r)
	os.Stdout = oldStdout
	expectedOutput := fmt.Sprintf("unsupported parameter type: %T for key %s", []int{1, 2, 3}, "param1")
	if !strings.Contains(string(out), expectedOutput) {
		t.Errorf("Expected output %q, got %q", expectedOutput, string(out))
	}
}
