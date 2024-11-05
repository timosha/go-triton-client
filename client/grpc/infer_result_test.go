package grpc

import (
	"errors"
	"github.com/Trendyol/go-triton-client/client/grpc/grpc_generated_v2"
	"github.com/Trendyol/go-triton-client/mocks"
	"testing"

	"github.com/golang/mock/gomock"
)

func TestNewInferResult_Success(t *testing.T) {
	mockController := gomock.NewController(t)
	defer mockController.Finish()

	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)

	response := &grpc_generated_v2.ModelInferResponse{
		ModelName:    "test_model",
		ModelVersion: "1",
		Outputs: []*grpc_generated_v2.ModelInferResponse_InferOutputTensor{
			{
				Name:     "output0",
				Datatype: "FP32",
				Shape:    []int64{2, 2},
			},
		},
		RawOutputContents: [][]byte{make([]byte, 16)},
	}

	mockResponseWrapper.EXPECT().GetResponse().Return(response).Times(1)

	mockDataConverter := mocks.NewMockDataConverter(mockController)

	result, err := NewInferResult(mockResponseWrapper, mockDataConverter, true)
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if result == nil {
		t.Error("Expected result, got nil")
	}
}

func TestNewInferResult_InvalidResponseType(t *testing.T) {
	mockController := gomock.NewController(t)
	defer mockController.Finish()

	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)

	mockResponseWrapper.EXPECT().GetResponse().Return("invalid_response").Times(1)

	mockDataConverter := mocks.NewMockDataConverter(mockController)

	result, err := NewInferResult(mockResponseWrapper, mockDataConverter, true)
	if err == nil {
		t.Error("Expected error, got nil")
	}
	if err.Error() != "invalid response type" {
		t.Errorf("Expected error 'invalid response type', got %v", err)
	}
	if result != nil {
		t.Error("Expected result to be nil")
	}
}

func TestNewInferResult_Verbose(t *testing.T) {
	mockController := gomock.NewController(t)
	defer mockController.Finish()

	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)

	response := &grpc_generated_v2.ModelInferResponse{
		ModelName:    "test_model",
		ModelVersion: "1",
		Outputs: []*grpc_generated_v2.ModelInferResponse_InferOutputTensor{
			{
				Name:     "output0",
				Datatype: "FP32",
				Shape:    []int64{2, 2},
			},
		},
		RawOutputContents: [][]byte{make([]byte, 16)},
	}

	mockResponseWrapper.EXPECT().GetResponse().Return(response).Times(1)

	mockDataConverter := mocks.NewMockDataConverter(mockController)

	result, err := NewInferResult(mockResponseWrapper, mockDataConverter, true)
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if result == nil {
		t.Error("Expected result, got nil")
	}
}

func TestNewInferResult_NoOutputs(t *testing.T) {
	mockController := gomock.NewController(t)
	defer mockController.Finish()

	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)

	response := &grpc_generated_v2.ModelInferResponse{
		ModelName:         "test_model",
		ModelVersion:      "1",
		Outputs:           []*grpc_generated_v2.ModelInferResponse_InferOutputTensor{},
		RawOutputContents: [][]byte{},
	}

	mockResponseWrapper.EXPECT().GetResponse().Return(response).Times(1)

	mockDataConverter := mocks.NewMockDataConverter(mockController)

	result, err := NewInferResult(mockResponseWrapper, mockDataConverter, true)
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if result == nil {
		t.Error("Expected result, got nil")
	}
}

func TestInferResult_AsSlice_Success(t *testing.T) {
	mockController := gomock.NewController(t)
	defer mockController.Finish()

	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)

	response := &grpc_generated_v2.ModelInferResponse{
		ModelName:    "test_model",
		ModelVersion: "1",
		Outputs: []*grpc_generated_v2.ModelInferResponse_InferOutputTensor{
			{
				Name:     "output0",
				Datatype: "FP32",
				Shape:    []int64{2, 2},
			},
		},
		RawOutputContents: [][]byte{make([]byte, 16)},
	}

	mockResponseWrapper.EXPECT().GetResponse().Return(response).Times(1)

	mockDataConverter := mocks.NewMockDataConverter(mockController)
	mockDataConverter.EXPECT().DeserializeTensor(gomock.Any(), gomock.Any()).Return([]float32{1, 2}, nil)
	mockDataConverter.EXPECT().ReshapeArray(gomock.Any(), gomock.Any()).Return([]interface{}{1, 2}, nil)

	resultInterface, err := NewInferResult(mockResponseWrapper, mockDataConverter, true)
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	result := resultInterface.(*InferResult)

	data, err := result.AsSlice("output0")
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if data == nil {
		t.Error("Expected data, got nil")
	}
}

func TestInferResult_AsSlice_OutputNotFound(t *testing.T) {
	mockController := gomock.NewController(t)
	defer mockController.Finish()

	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)

	response := &grpc_generated_v2.ModelInferResponse{
		ModelName:    "test_model",
		ModelVersion: "1",
		Outputs: []*grpc_generated_v2.ModelInferResponse_InferOutputTensor{
			{
				Name:       "output1",
				Datatype:   "FP32",
				Shape:      []int64{1, 1},
				Parameters: nil,
				Contents:   nil,
			},
		},
		RawOutputContents: [][]byte{[]byte("test")},
	}

	mockResponseWrapper.EXPECT().GetResponse().Return(response).Times(1)

	mockDataConverter := mocks.NewMockDataConverter(mockController)

	resultInterface, err := NewInferResult(mockResponseWrapper, mockDataConverter, true)
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	result := resultInterface.(*InferResult)

	_, err = result.AsSlice("output0")
	if err == nil {
		t.Error("Expected error, got nil")
	}
	if err.Error() != "output output0 not found" {
		t.Errorf("Expected error 'output output0 not found', got %v", err)
	}
}

func TestInferResult_AsSlice_DeserializeError(t *testing.T) {
	mockController := gomock.NewController(t)
	defer mockController.Finish()

	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)

	response := &grpc_generated_v2.ModelInferResponse{
		ModelName:    "test_model",
		ModelVersion: "1",
		Outputs: []*grpc_generated_v2.ModelInferResponse_InferOutputTensor{
			{
				Name:     "output0",
				Datatype: "FP32",
				Shape:    []int64{1, 2},
			},
		},
		RawOutputContents: [][]byte{make([]byte, 16)},
	}

	mockResponseWrapper.EXPECT().GetResponse().Return(response).Times(1)

	mockDataConverter := mocks.NewMockDataConverter(mockController)
	mockDataConverter.EXPECT().DeserializeTensor(gomock.Any(), gomock.Any()).Return(nil, errors.New("deserialize error"))

	resultInterface, err := NewInferResult(mockResponseWrapper, mockDataConverter, true)
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	result := resultInterface.(*InferResult)

	_, err = result.AsSlice("output0")
	if err == nil {
		t.Error("Expected error, got nil")
	}
	if err.Error() != "deserialize error" {
		t.Errorf("Expected error 'deserialize error', got %v", err)
	}
}

func TestInferResult_AsSlice_ReshapeError(t *testing.T) {
	mockController := gomock.NewController(t)
	defer mockController.Finish()

	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)

	response := &grpc_generated_v2.ModelInferResponse{
		ModelName:    "test_model",
		ModelVersion: "1",
		Outputs: []*grpc_generated_v2.ModelInferResponse_InferOutputTensor{
			{
				Name:     "output0",
				Datatype: "FP32",
				Shape:    []int64{2, 2},
			},
		},
		RawOutputContents: [][]byte{make([]byte, 16)},
	}

	mockResponseWrapper.EXPECT().GetResponse().Return(response).Times(1)

	mockDataConverter := mocks.NewMockDataConverter(mockController)
	mockDataConverter.EXPECT().DeserializeTensor(gomock.Any(), gomock.Any()).Return([]float32{1, 2}, nil)
	mockDataConverter.EXPECT().ReshapeArray(gomock.Any(), gomock.Any()).Return(nil, errors.New("reshape error"))

	resultInterface, err := NewInferResult(mockResponseWrapper, mockDataConverter, true)
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	result := resultInterface.(*InferResult)

	_, err = result.AsSlice("output0")
	if err == nil {
		t.Error("Expected error, got nil")
	}
	if err.Error() != "reshape error" {
		t.Errorf("Expected error 'reshape error', got %v", err)
	}
}

func TestInferResult_GetOutput_Found(t *testing.T) {
	mockController := gomock.NewController(t)
	defer mockController.Finish()

	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)

	response := &grpc_generated_v2.ModelInferResponse{
		ModelName:    "test_model",
		ModelVersion: "1",
		Outputs: []*grpc_generated_v2.ModelInferResponse_InferOutputTensor{
			{
				Name:     "output0",
				Datatype: "FP32",
				Shape:    []int64{2, 2},
			},
		},
		RawOutputContents: [][]byte{make([]byte, 16)},
	}

	mockResponseWrapper.EXPECT().GetResponse().Return(response).Times(1)

	mockDataConverter := mocks.NewMockDataConverter(mockController)

	resultInterface, err := NewInferResult(mockResponseWrapper, mockDataConverter, true)
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	result := resultInterface.(*InferResult)

	output, err := result.GetOutput("output0")
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if output.GetName() != "output0" {
		t.Errorf("Expected output name 'output0', got '%s'", output.GetName())
	}
}

func TestInferResult_GetOutput_NotFound(t *testing.T) {
	mockController := gomock.NewController(t)
	defer mockController.Finish()

	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)

	response := &grpc_generated_v2.ModelInferResponse{
		ModelName:    "test_model",
		ModelVersion: "1",
		Outputs: []*grpc_generated_v2.ModelInferResponse_InferOutputTensor{
			{
				Name:       "output1",
				Datatype:   "FP32",
				Shape:      []int64{1, 1},
				Parameters: nil,
				Contents:   nil,
			},
		},
		RawOutputContents: [][]byte{[]byte("test")},
	}

	mockResponseWrapper.EXPECT().GetResponse().Return(response).Times(1)

	mockDataConverter := mocks.NewMockDataConverter(mockController)

	resultInterface, err := NewInferResult(mockResponseWrapper, mockDataConverter, true)
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	result := resultInterface.(*InferResult)

	_, err = result.GetOutput("output0")
	if err == nil {
		t.Error("Expected error, got nil")
	}
	if err.Error() != "output output0 not found" {
		t.Errorf("Expected error 'output output0 not found', got %v", err)
	}
}

func TestConvertInt64ToInt(t *testing.T) {
	int64Slice := []int64{1, 2, 3}
	intSlice := convertInt64ToInt(int64Slice)
	if len(intSlice) != len(int64Slice) {
		t.Errorf("Expected length %d, got %d", len(int64Slice), len(intSlice))
	}
	for i, v := range intSlice {
		if v != int(int64Slice[i]) {
			t.Errorf("Expected value %d, got %d", int(int64Slice[i]), v)
		}
	}
}

func TestInferResult_AsSlice_MissingBufferMapEntry(t *testing.T) {
	mockController := gomock.NewController(t)
	defer mockController.Finish()

	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)

	response := &grpc_generated_v2.ModelInferResponse{
		ModelName:    "test_model",
		ModelVersion: "1",
		Outputs: []*grpc_generated_v2.ModelInferResponse_InferOutputTensor{
			{
				Name:     "output1",
				Datatype: "FP32",
				Shape:    []int64{2, 2},
			},
		},
		RawOutputContents: [][]byte{make([]byte, 16)},
	}

	mockResponseWrapper.EXPECT().GetResponse().Return(response).Times(1)

	mockDataConverter := mocks.NewMockDataConverter(mockController)

	resultInterface, err := NewInferResult(mockResponseWrapper, mockDataConverter, true)
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	result := resultInterface.(*InferResult)

	_, err = result.AsSlice("output0")
	if err == nil {
		t.Error("Expected error, got nil")
	}
	if err.Error() != "output output0 not found" {
		t.Errorf("Expected error 'output output0 not found', got %v", err)
	}
}
