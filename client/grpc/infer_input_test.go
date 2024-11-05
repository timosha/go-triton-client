package grpc

import (
	"github.com/Trendyol/go-triton-client/base"
	"github.com/Trendyol/go-triton-client/client/grpc/grpc_generated_v2"
	"github.com/Trendyol/go-triton-client/mocks"
	"github.com/golang/mock/gomock"
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

func TestInferInput_GetTensor_WithRawData(t *testing.T) {
	name := "input0"
	datatype := "FP32"
	shape := []int64{1, 3}
	rawData := []byte{1, 2, 3, 4}
	mockController := gomock.NewController(t)
	defer mockController.Finish()
	dataConverter := mocks.NewMockDataConverter(mockController)
	input := &InferInput{
		BaseInferInput: &base.BaseInferInput{
			Name:          name,
			Datatype:      datatype,
			Shape:         shape,
			RawData:       rawData,
			DataConverter: dataConverter,
		},
	}
	tensor := input.GetTensor().(*grpc_generated_v2.ModelInferRequest_InferInputTensor)
	if tensor.Name != name {
		t.Errorf("Expected Name %s, got %s", name, tensor.Name)
	}
	if tensor.Datatype != datatype {
		t.Errorf("Expected Datatype %s, got %s", datatype, tensor.Datatype)
	}
	if !reflect.DeepEqual(tensor.Shape, shape) {
		t.Errorf("Expected Shape %v, got %v", shape, tensor.Shape)
	}
	if tensor.Contents != nil {
		t.Errorf("Expected Contents to be nil when RawData is present")
	}
}

func TestInferInput_GetTensor_WithContents(t *testing.T) {
	name := "input0"
	datatype := "FP32"
	shape := []int64{1, 3}
	mockController := gomock.NewController(t)
	defer mockController.Finish()
	dataConverter := mocks.NewMockDataConverter(mockController)
	dataConverter.EXPECT().ConvertByteSliceToFloat32Slice(gomock.Any()).Return([]float32{1.1, 2.2})
	input := &InferInput{
		BaseInferInput: &base.BaseInferInput{
			Name:          name,
			Datatype:      datatype,
			Shape:         shape,
			RawData:       nil,
			DataConverter: dataConverter,
		},
	}
	tensor := input.GetTensor().(*grpc_generated_v2.ModelInferRequest_InferInputTensor)
	if tensor.Name != name {
		t.Errorf("Expected Name %s, got %s", name, tensor.Name)
	}
	if tensor.Datatype != datatype {
		t.Errorf("Expected Datatype %s, got %s", datatype, tensor.Datatype)
	}
	if !reflect.DeepEqual(tensor.Shape, shape) {
		t.Errorf("Expected Shape %v, got %v", shape, tensor.Shape)
	}
	if tensor.Contents == nil {
		t.Fatalf("Expected Contents to be non-nil")
	}
	expectedFp32Contents := []float32{1.1, 2.2}
	if !reflect.DeepEqual(tensor.Contents.Fp32Contents, expectedFp32Contents) {
		t.Errorf("Expected Fp32Contents %v, got %v", expectedFp32Contents, tensor.Contents.Fp32Contents)
	}
}

func TestInferInput_GetTensor_WithContentsFP64(t *testing.T) {
	name := "input0"
	datatype := "FP64"
	shape := []int64{1, 3}
	mockController := gomock.NewController(t)
	defer mockController.Finish()
	dataConverter := mocks.NewMockDataConverter(mockController)
	dataConverter.EXPECT().ConvertByteSliceToFloat64Slice(gomock.Any()).Return([]float64{1.1, 2.2})
	input := &InferInput{
		BaseInferInput: &base.BaseInferInput{
			Name:          name,
			Datatype:      datatype,
			Shape:         shape,
			RawData:       nil,
			DataConverter: dataConverter,
		},
	}
	tensor := input.GetTensor().(*grpc_generated_v2.ModelInferRequest_InferInputTensor)
	if tensor.Name != name {
		t.Errorf("Expected Name %s, got %s", name, tensor.Name)
	}
	if tensor.Datatype != datatype {
		t.Errorf("Expected Datatype %s, got %s", datatype, tensor.Datatype)
	}
	if !reflect.DeepEqual(tensor.Shape, shape) {
		t.Errorf("Expected Shape %v, got %v", shape, tensor.Shape)
	}
	if tensor.Contents == nil {
		t.Fatalf("Expected Contents to be non-nil")
	}
	expectedFp64Contents := []float64{1.1, 2.2}
	if !reflect.DeepEqual(tensor.Contents.Fp64Contents, expectedFp64Contents) {
		t.Errorf("Expected Fp64Contents %v, got %v", expectedFp64Contents, tensor.Contents.Fp64Contents)
	}
}

func TestInferInput_GetTensor_WithContentsINT64(t *testing.T) {
	name := "input0"
	datatype := "INT64"
	shape := []int64{1, 3}
	mockController := gomock.NewController(t)
	defer mockController.Finish()
	dataConverter := mocks.NewMockDataConverter(mockController)
	dataConverter.EXPECT().ConvertByteSliceToInt64Slice(gomock.Any()).Return([]int64{1, 2})
	input := &InferInput{
		BaseInferInput: &base.BaseInferInput{
			Name:          name,
			Datatype:      datatype,
			Shape:         shape,
			RawData:       nil,
			DataConverter: dataConverter,
		},
	}
	tensor := input.GetTensor().(*grpc_generated_v2.ModelInferRequest_InferInputTensor)
	if tensor.Name != name {
		t.Errorf("Expected Name %s, got %s", name, tensor.Name)
	}
	if tensor.Datatype != datatype {
		t.Errorf("Expected Datatype %s, got %s", datatype, tensor.Datatype)
	}
	if !reflect.DeepEqual(tensor.Shape, shape) {
		t.Errorf("Expected Shape %v, got %v", shape, tensor.Shape)
	}
	if tensor.Contents == nil {
		t.Fatalf("Expected Contents to be non-nil")
	}
	expectedContents := []int64{1, 2}
	if !reflect.DeepEqual(tensor.Contents.Int64Contents, expectedContents) {
		t.Errorf("Expected INT64Contents %v, got %v", expectedContents, tensor.Contents.Int64Contents)
	}
}

func TestInferInput_GetTensor_WithContentsBYTES(t *testing.T) {
	name := "input0"
	datatype := "BYTES"
	shape := []int64{1, 3}
	mockController := gomock.NewController(t)
	defer mockController.Finish()
	dataConverter := mocks.NewMockDataConverter(mockController)
	input := &InferInput{
		BaseInferInput: &base.BaseInferInput{
			Name:          name,
			Datatype:      datatype,
			Shape:         shape,
			RawData:       nil,
			DataConverter: dataConverter,
		},
	}
	tensor := input.GetTensor().(*grpc_generated_v2.ModelInferRequest_InferInputTensor)
	if tensor.Name != name {
		t.Errorf("Expected Name %s, got %s", name, tensor.Name)
	}
	if tensor.Datatype != datatype {
		t.Errorf("Expected Datatype %s, got %s", datatype, tensor.Datatype)
	}
	if !reflect.DeepEqual(tensor.Shape, shape) {
		t.Errorf("Expected Shape %v, got %v", shape, tensor.Shape)
	}
	if tensor.Contents == nil {
		t.Fatalf("Expected Contents to be non-nil")
	}

	expectedBytesContents := [][]byte{nil}
	if !reflect.DeepEqual(tensor.Contents.BytesContents, expectedBytesContents) {
		t.Errorf("Expected BytesContents %v, got %v", expectedBytesContents, tensor.Contents.BytesContents)
	}
}

func TestInferInput_GetTensor_UnsupportedDatatype(t *testing.T) {
	name := "input0"
	datatype := "UNKNOWN"
	shape := []int64{1, 3}
	mockController := gomock.NewController(t)
	defer mockController.Finish()
	dataConverter := mocks.NewMockDataConverter(mockController)
	input := &InferInput{
		BaseInferInput: &base.BaseInferInput{
			Name:          name,
			Datatype:      datatype,
			Shape:         shape,
			RawData:       nil,
			DataConverter: dataConverter,
		},
	}

	tensor := input.GetTensor().(*grpc_generated_v2.ModelInferRequest_InferInputTensor)
	if tensor.Contents != nil {
		t.Errorf("Expected Contents to be nil for unsupported datatype")
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
