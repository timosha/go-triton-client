package base

import (
	"errors"
	"github.com/Trendyol/go-triton-client/mocks"
	"github.com/golang/mock/gomock"
	"reflect"
	"testing"
)

func TestBaseInferInput_GetName(t *testing.T) {
	input := &BaseInferInput{Name: "input0"}
	if input.GetName() != "input0" {
		t.Errorf("Expected Name 'input0', got %s", input.GetName())
	}
}

func TestBaseInferInput_GetShape(t *testing.T) {
	shape := []int64{1, 3}
	input := &BaseInferInput{Shape: shape}
	if !reflect.DeepEqual(input.GetShape(), shape) {
		t.Errorf("Expected Shape %v, got %v", shape, input.GetShape())
	}
}

func TestBaseInferInput_GetDatatype(t *testing.T) {
	input := &BaseInferInput{Datatype: "FP32"}
	if input.GetDatatype() != "FP32" {
		t.Errorf("Expected Datatype 'FP32', got %s", input.GetDatatype())
	}
}

func TestBaseInferInput_GetParameters(t *testing.T) {
	parameters := map[string]interface{}{"param1": "value1"}
	input := &BaseInferInput{Parameters: parameters}
	if !reflect.DeepEqual(input.GetParameters(), parameters) {
		t.Errorf("Expected Parameters %v, got %v", parameters, input.GetParameters())
	}
}

func TestBaseInferInput_GetData(t *testing.T) {
	data := []interface{}{1, 2, 3}
	input := &BaseInferInput{Data: data}
	if !reflect.DeepEqual(input.GetData(), data) {
		t.Errorf("Expected Data %v, got %v", data, input.GetData())
	}
}

func TestBaseInferInput_GetRawData(t *testing.T) {
	rawData := []byte{1, 2, 3, 4}
	input := &BaseInferInput{RawData: rawData}
	if !reflect.DeepEqual(input.GetRawData(), rawData) {
		t.Errorf("Expected RawData %v, got %v", rawData, input.GetRawData())
	}
}

func TestBaseInferInput_SetData(t *testing.T) {
	inputTensor := []int32{1, 2, 3}
	datatype := "INT32"
	mockController := gomock.NewController(t)
	defer mockController.Finish()
	dataConverter := mocks.NewMockDataConverter(mockController)
	dataConverter.EXPECT().SerializeTensor(gomock.Any()).Return([]uint8{1, 2, 3}, nil)
	input := &BaseInferInput{
		Datatype:      datatype,
		Parameters:    map[string]interface{}{"binary_data_size": 100},
		DataConverter: dataConverter,
	}
	err := input.SetData(inputTensor, true)
	if err != nil {
		t.Fatalf("SetData returned error: %v", err)
	}
	if _, ok := input.Parameters["binary_data_size"]; !ok {
		t.Error("Expected 'binary_data_size' parameter to be not deleted")
	}
	if input.Data != nil {
		t.Errorf("Expected RawData to be nil, got %v", input.RawData)
	}
	expectedData := []uint8{1, 2, 3}
	if !reflect.DeepEqual(input.RawData, expectedData) {
		t.Errorf("Expected Data %v, got %v", expectedData, input.Data)
	}
}

func TestBaseInferInput_SetData_NonBinaryData(t *testing.T) {
	inputTensor := []int32{1, 2, 3}
	datatype := "INT32"
	mockController := gomock.NewController(t)
	defer mockController.Finish()
	dataConverter := mocks.NewMockDataConverter(mockController)
	dataConverter.EXPECT().FlattenData(gomock.Any()).Return([]interface{}{int32(1), int32(2), int32(3)})
	input := &BaseInferInput{
		Datatype:      datatype,
		Parameters:    map[string]interface{}{"binary_data_size": 100},
		DataConverter: dataConverter,
	}
	err := input.SetData(inputTensor, false)
	if err != nil {
		t.Fatalf("SetData returned error: %v", err)
	}
	if _, ok := input.Parameters["binary_data_size"]; ok {
		t.Error("Expected 'binary_data_size' parameter to be deleted")
	}
	if input.RawData != nil {
		t.Errorf("Expected RawData to be nil, got %v", input.RawData)
	}
	expectedData := []interface{}{int32(1), int32(2), int32(3)}
	if !reflect.DeepEqual(input.Data, expectedData) {
		t.Errorf("Expected Data %v, got %v", expectedData, input.Data)
	}
}

func TestBaseInferInput_SetData_BinaryData(t *testing.T) {
	inputTensor := []int32{1, 2, 3}
	datatype := "INT32"
	mockController := gomock.NewController(t)
	defer mockController.Finish()
	dataConverter := mocks.NewMockDataConverter(mockController)
	dataConverter.EXPECT().SerializeTensor(gomock.Any()).Return([]byte{1, 2, 3, 4}, nil)
	input := &BaseInferInput{
		Datatype:      datatype,
		Parameters:    make(map[string]interface{}),
		DataConverter: dataConverter,
	}
	err := input.SetData(inputTensor, true)
	if err != nil {
		t.Fatalf("SetData returned error: %v", err)
	}
	if input.Data != nil {
		t.Errorf("Expected Data to be nil, got %v", input.Data)
	}
	expectedRawData := []byte{1, 2, 3, 4}
	if !reflect.DeepEqual(input.RawData, expectedRawData) {
		t.Errorf("Expected RawData %v, got %v", expectedRawData, input.RawData)
	}
	if size, ok := input.Parameters["binary_data_size"]; !ok || size != len(expectedRawData) {
		t.Errorf("Expected binary_data_size %d, got %v", len(expectedRawData), size)
	}
}

func TestBaseInferInput_SetData_InvalidDatatype(t *testing.T) {
	inputTensor := []float32{1.0, 2.0, 3.0}
	datatype := "INT32"
	mockController := gomock.NewController(t)
	defer mockController.Finish()
	dataConverter := mocks.NewMockDataConverter(mockController)
	input := &BaseInferInput{
		Datatype:      datatype,
		Parameters:    make(map[string]interface{}),
		DataConverter: dataConverter,
	}
	err := input.SetData(inputTensor, false)
	if err == nil {
		t.Fatal("Expected error due to mismatched datatype, got nil")
	}
}

func TestBaseInferInput_SetData_SerializeError(t *testing.T) {
	inputTensor := []string{"a", "b", "c"}
	datatype := "BYTES"
	mockController := gomock.NewController(t)
	defer mockController.Finish()
	dataConverter := mocks.NewMockDataConverter(mockController)
	dataConverter.EXPECT().SerializeTensor(gomock.Any()).Return(nil, errors.New("error while SerializeTensor"))
	input := &BaseInferInput{
		Datatype:      datatype,
		Parameters:    make(map[string]interface{}),
		DataConverter: dataConverter,
	}
	err := input.SetData(inputTensor, true)
	if err == nil {
		t.Fatal("Expected error from SerializeTensor, got nil")
	}
}

func TestGetDatatype(t *testing.T) {
	tests := []struct {
		input    interface{}
		expected string
	}{
		{[]int{1, 2}, "INT64"},
		{[]int32{1, 2}, "INT32"},
		{[]int64{1, 2}, "INT64"},
		{[]uint16{1, 2}, "UINT16"},
		{[]uint32{1, 2}, "UINT32"},
		{[]uint64{1, 2}, "UINT64"},
		{[]float32{1.0, 2.0}, "FP32"},
		{[]float64{1.0, 2.0}, "FP64"},
		{[]byte{1, 2}, "BYTES"},
		{[]bool{true, false}, "BOOL"},
		{[]string{"a", "b"}, "BYTES"},
		{[]complex64{1 + 2i}, "UNKNOWN"},
	}
	for _, tt := range tests {
		datatype := getDatatype(tt.input)
		if datatype != tt.expected {
			t.Errorf("getDatatype(%T) = %s, expected %s", tt.input, datatype, tt.expected)
		}
	}
}

func TestBaseInferInput_SetShape(t *testing.T) {
	shape := []int64{1, 3}
	input := &BaseInferInput{}
	input.SetShape(shape)
	if !reflect.DeepEqual(input.GetShape(), shape) {
		t.Errorf("Expected Shape %v, got %v", shape, input.GetShape())
	}
}

func TestBaseInferInput_SetDatatype(t *testing.T) {
	input := &BaseInferInput{}
	input.SetDatatype("FP32")
	if input.GetDatatype() != "FP32" {
		t.Errorf("Expected Datatype 'FP32', got %s", input.GetDatatype())
	}
}
