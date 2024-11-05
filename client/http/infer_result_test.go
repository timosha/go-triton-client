package http

import (
	"bytes"
	"compress/gzip"
	"compress/zlib"
	"errors"
	"github.com/Trendyol/go-triton-client/mocks"
	"github.com/golang/mock/gomock"
	"strconv"
	"testing"
)

func TestNewInferResult_NoHeaderLength(t *testing.T) {
	body := []byte(`{"model_name":"test_model","model_version":"1","outputs":[{"name":"output0","datatype":"FP32","shape":[2,2]}]}`)
	mockController := gomock.NewController(t)
	defer mockController.Finish()
	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)
	mockResponseWrapper.EXPECT().GetHeader("Inference-Header-Content-Length").Return("")
	mockResponseWrapper.EXPECT().GetHeader("Content-Encoding").Return("")
	mockResponseWrapper.EXPECT().GetBody().Return(body, nil)
	mockDataConverter := mocks.NewMockDataConverter(mockController)
	result, err := NewInferResult(mockResponseWrapper, mockDataConverter, true)
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if result == nil {
		t.Error("Expected result, got nil")
	}
}

func TestNewInferResult_WithHeaderLength(t *testing.T) {
	body := []byte(`{"model_name":"test_model","model_version":"1","outputs":[{"name":"output0","datatype":"FP32","shape":[2,2],"parameters":{"binary_data_size":16}}]}`)
	mockController := gomock.NewController(t)
	defer mockController.Finish()
	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)
	mockResponseWrapper.EXPECT().GetHeader("Inference-Header-Content-Length").Return("")
	mockResponseWrapper.EXPECT().GetHeader("Content-Encoding").Return("")
	mockResponseWrapper.EXPECT().GetBody().Return(body, nil)
	mockDataConverter := mocks.NewMockDataConverter(mockController)
	result, err := NewInferResult(mockResponseWrapper, mockDataConverter, true)
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if result == nil {
		t.Error("Expected result, got nil")
	}
}

func TestNewInferResult_UnmarshalError(t *testing.T) {
	body := []byte(`{"invalid_json"`)
	mockController := gomock.NewController(t)
	defer mockController.Finish()
	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)
	mockResponseWrapper.EXPECT().GetHeader("Inference-Header-Content-Length").Return(strconv.Itoa(len(body)))
	mockResponseWrapper.EXPECT().GetHeader("Content-Encoding").Return("")
	mockResponseWrapper.EXPECT().GetBody().Return(body, nil)
	mockDataConverter := mocks.NewMockDataConverter(mockController)
	_, err := NewInferResult(mockResponseWrapper, mockDataConverter, true)
	if err == nil {
		t.Error("Expected error, got nil")
	}
}

func TestNewInferResult_Gzip(t *testing.T) {
	var buf bytes.Buffer
	gw := gzip.NewWriter(&buf)
	body := []byte(`{"model_name":"test_model","model_version":"1","outputs":[]}`)
	gw.Write(body)
	gw.Close()
	mockController := gomock.NewController(t)
	defer mockController.Finish()
	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)
	mockResponseWrapper.EXPECT().GetHeader("Inference-Header-Content-Length").Return("")
	mockResponseWrapper.EXPECT().GetHeader("Content-Encoding").Return("")
	mockResponseWrapper.EXPECT().GetBody().Return(body, nil)
	mockDataConverter := mocks.NewMockDataConverter(mockController)
	result, err := NewInferResult(mockResponseWrapper, mockDataConverter, true)
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if result == nil {
		t.Error("Expected result, got nil")
	}
}

func TestNewInferResult_Deflate(t *testing.T) {
	var buf bytes.Buffer
	zw := zlib.NewWriter(&buf)
	body := []byte(`{"model_name":"test_model","model_version":"1","outputs":[]}`)
	zw.Write(body)
	zw.Close()
	mockController := gomock.NewController(t)
	defer mockController.Finish()
	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)
	mockResponseWrapper.EXPECT().GetHeader("Inference-Header-Content-Length").Return("")
	mockResponseWrapper.EXPECT().GetHeader("Content-Encoding").Return("")
	mockResponseWrapper.EXPECT().GetBody().Return(body, nil)
	mockDataConverter := mocks.NewMockDataConverter(mockController)
	result, err := NewInferResult(mockResponseWrapper, mockDataConverter, true)
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if result == nil {
		t.Error("Expected result, got nil")
	}
}

func TestNewInferResult_InvalidContentEncoding(t *testing.T) {
	mockController := gomock.NewController(t)
	defer mockController.Finish()
	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)
	mockResponseWrapper.EXPECT().GetHeader("Inference-Header-Content-Length").Return("")
	mockResponseWrapper.EXPECT().GetHeader("Content-Encoding").Return("invalid")
	mockResponseWrapper.EXPECT().GetBody().Return([]byte(`data`), nil)
	mockDataConverter := mocks.NewMockDataConverter(mockController)
	_, err := NewInferResult(mockResponseWrapper, mockDataConverter, true)
	if err == nil {
		t.Errorf("Expected error")
	}
}

func TestInferResult_AsSlice_HasBinaryData(t *testing.T) {
	body := []byte(`{"model_name":"test_model","model_version":"1","outputs":[{"name":"output0","datatype":"FP32","shape":[2,2],"parameters":{"binary_data_size":16}}]}`)
	mockController := gomock.NewController(t)
	defer mockController.Finish()
	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)
	mockResponseWrapper.EXPECT().GetHeader("Inference-Header-Content-Length").Return(strconv.Itoa(len(body)))
	mockResponseWrapper.EXPECT().GetHeader("Content-Encoding").Return("")
	mockResponseWrapper.EXPECT().GetBody().Return(append(body, make([]byte, 16)...), nil)
	mockDataConverter := mocks.NewMockDataConverter(mockController)
	mockDataConverter.EXPECT().DeserializeTensor(gomock.Any(), gomock.Any()).Return([]float32{1, 2}, nil)
	mockDataConverter.EXPECT().ReshapeArray(gomock.Any(), gomock.Any()).Return([]interface{}{1, 2}, nil)
	result, _ := NewInferResult(mockResponseWrapper, mockDataConverter, true)
	data, err := result.AsSlice("output0")
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if data == nil {
		t.Error("Expected data, got nil")
	}
}

func TestInferResult_AsSlice_NoBinaryData(t *testing.T) {
	body := []byte(`{"model_name":"test_model","model_version":"1","outputs":[{"name":"output0","datatype":"FP32","shape":[2,2],"data":[1.0,2.0,3.0,4.0]}]}`)
	mockController := gomock.NewController(t)
	defer mockController.Finish()
	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)
	mockResponseWrapper.EXPECT().GetHeader("Inference-Header-Content-Length").Return("")
	mockResponseWrapper.EXPECT().GetHeader("Content-Encoding").Return("")
	mockResponseWrapper.EXPECT().GetBody().Return(body, nil)
	mockDataConverter := mocks.NewMockDataConverter(mockController)
	mockDataConverter.EXPECT().ReshapeArray(gomock.Any(), gomock.Any()).Return([]interface{}{1, 2}, nil)
	result, _ := NewInferResult(mockResponseWrapper, mockDataConverter, true)
	data, err := result.AsSlice("output0")
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if data == nil {
		t.Error("Expected data, got nil")
	}
}

func TestInferResult_GetOutput_NotFound(t *testing.T) {
	body := []byte(`{"model_name":"test_model","model_version":"1","outputs":[{"name":"output1","datatype":"FP32","shape":[2,2]}]}`)
	mockController := gomock.NewController(t)
	defer mockController.Finish()
	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)
	mockResponseWrapper.EXPECT().GetHeader("Inference-Header-Content-Length").Return("")
	mockResponseWrapper.EXPECT().GetHeader("Content-Encoding").Return("")
	mockResponseWrapper.EXPECT().GetBody().Return(body, nil)
	mockDataConverter := mocks.NewMockDataConverter(mockController)
	result, _ := NewInferResult(mockResponseWrapper, mockDataConverter, true)
	_, err := result.GetOutput("output0")
	if err == nil || err.Error() != "output output0 not found" {
		t.Errorf("Expected error 'output output0 not found', got %v", err)
	}
}

func TestInferResult_AsSlice_DeserializeError(t *testing.T) {
	body := []byte(`{"model_name":"test_model","model_version":"1","outputs":[{"name":"output0","datatype":"FP32","shape":[2,2],"parameters":{"binary_data_size":16}}]}`)
	mockController := gomock.NewController(t)
	defer mockController.Finish()
	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)
	mockResponseWrapper.EXPECT().GetHeader("Inference-Header-Content-Length").Return(strconv.Itoa(len(body)))
	mockResponseWrapper.EXPECT().GetHeader("Content-Encoding").Return("")
	mockResponseWrapper.EXPECT().GetBody().Return(append(body, make([]byte, 16)...), nil)
	mockDataConverter := mocks.NewMockDataConverter(mockController)
	mockDataConverter.EXPECT().DeserializeTensor(gomock.Any(), gomock.Any()).Return(nil, errors.New("deserialize error"))
	result, _ := NewInferResult(mockResponseWrapper, mockDataConverter, true)
	_, err := result.AsSlice("output0")
	if err == nil || err.Error() != "deserialize error" {
		t.Errorf("Expected error 'deserialize error', got %v", err)
	}
}

func TestInferResult_AsSlice_ReshapeError(t *testing.T) {
	body := []byte(`{"model_name":"test_model","model_version":"1","outputs":[{"name":"output0","datatype":"FP32","shape":[2,2],"data":[1.0,2.0,3.0,4.0]}]}`)
	mockController := gomock.NewController(t)
	defer mockController.Finish()
	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)
	mockResponseWrapper.EXPECT().GetHeader("Inference-Header-Content-Length").Return("")
	mockResponseWrapper.EXPECT().GetHeader("Content-Encoding").Return("")
	mockResponseWrapper.EXPECT().GetBody().Return(body, nil)
	mockDataConverter := mocks.NewMockDataConverter(mockController)
	mockDataConverter.EXPECT().ReshapeArray(gomock.Any(), gomock.Any()).Return(nil, errors.New("reshape error"))
	result, _ := NewInferResult(mockResponseWrapper, mockDataConverter, true)
	_, err := result.AsSlice("output0")
	if err == nil || err.Error() != "reshape error" {
		t.Errorf("Expected error 'reshape error', got %v", err)
	}
}

func TestInferResult_AsSlice_NotFound(t *testing.T) {
	body := []byte(`{"model_name":"test_model","model_version":"1","outputs":[{"name":"output0","datatype":"FP32","shape":[2,2],"parameters":{"binary_data_size":16}}]}`)
	mockController := gomock.NewController(t)
	defer mockController.Finish()
	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)
	mockResponseWrapper.EXPECT().GetHeader("Inference-Header-Content-Length").Return(strconv.Itoa(len(body)))
	mockResponseWrapper.EXPECT().GetHeader("Content-Encoding").Return("")
	mockResponseWrapper.EXPECT().GetBody().Return(append(body, make([]byte, 16)...), nil)
	mockDataConverter := mocks.NewMockDataConverter(mockController)
	result, _ := NewInferResult(mockResponseWrapper, mockDataConverter, true)
	_, err := result.AsSlice("output1")
	if err == nil || err.Error() != "output output1 not found" {
		t.Errorf("Expected error 'output output1 not found', got %v", err)
	}
}

func TestNewInferResult_BodyError(t *testing.T) {
	mockController := gomock.NewController(t)
	defer mockController.Finish()
	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)
	mockResponseWrapper.EXPECT().GetHeader("Inference-Header-Content-Length").Return("")
	mockResponseWrapper.EXPECT().GetHeader("Content-Encoding").Return("")
	mockResponseWrapper.EXPECT().GetBody().Return(nil, errors.New("body error"))
	mockDataConverter := mocks.NewMockDataConverter(mockController)
	_, err := NewInferResult(mockResponseWrapper, mockDataConverter, true)
	if err == nil || err.Error() != "body error" {
		t.Errorf("Expected error 'body error', got %v", err)
	}
}

func TestNewInferResult_Deflate_ReadAllError(t *testing.T) {
	var buf bytes.Buffer
	zw := zlib.NewWriter(&buf)
	zw.Write([]byte(`{"model_name":"test_model","model_version":"1","outputs":[]}`))
	zw.Close()
	faultyData := buf.Bytes()[:10]
	mockController := gomock.NewController(t)
	defer mockController.Finish()
	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)
	mockResponseWrapper.EXPECT().GetHeader("Inference-Header-Content-Length").Return("")
	mockResponseWrapper.EXPECT().GetHeader("Content-Encoding").Return("deflate")
	mockResponseWrapper.EXPECT().GetBody().Return(faultyData, nil)
	mockDataConverter := mocks.NewMockDataConverter(mockController)
	_, err := NewInferResult(mockResponseWrapper, mockDataConverter, true)
	if err == nil {
		t.Error("Expected error, got nil")
	}
}

func TestNewInferResult_Gzip_ReadAllError(t *testing.T) {
	var buf bytes.Buffer
	gw := gzip.NewWriter(&buf)
	gw.Write([]byte(`{"model_name":"test_model","model_version":"1","outputs":[]}`))
	gw.Close()
	faultyData := buf.Bytes()[:10]
	mockController := gomock.NewController(t)
	defer mockController.Finish()
	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)
	mockResponseWrapper.EXPECT().GetHeader("Inference-Header-Content-Length").Return("")
	mockResponseWrapper.EXPECT().GetHeader("Content-Encoding").Return("gzip")
	mockResponseWrapper.EXPECT().GetBody().Return(faultyData, nil)
	mockDataConverter := mocks.NewMockDataConverter(mockController)
	_, err := NewInferResult(mockResponseWrapper, mockDataConverter, true)
	if err == nil {
		t.Error("Expected error, got nil")
	}
}

func TestNewInferResult_GzipReaderError(t *testing.T) {
	mockController := gomock.NewController(t)
	defer mockController.Finish()
	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)
	mockResponseWrapper.EXPECT().GetHeader("Inference-Header-Content-Length").Return("")
	mockResponseWrapper.EXPECT().GetHeader("Content-Encoding").Return("gzip")
	mockResponseWrapper.EXPECT().GetBody().Return([]byte("invalid gzip data"), nil)
	mockDataConverter := mocks.NewMockDataConverter(mockController)
	_, err := NewInferResult(mockResponseWrapper, mockDataConverter, true)
	if err == nil {
		t.Error("Expected error, got nil")
	}
}

func TestNewInferResult_ZlibReaderError(t *testing.T) {
	mockController := gomock.NewController(t)
	defer mockController.Finish()
	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)
	mockResponseWrapper.EXPECT().GetHeader("Inference-Header-Content-Length").Return("")
	mockResponseWrapper.EXPECT().GetHeader("Content-Encoding").Return("deflate")
	mockResponseWrapper.EXPECT().GetBody().Return([]byte("invalid deflate data"), nil)
	mockDataConverter := mocks.NewMockDataConverter(mockController)
	_, err := NewInferResult(mockResponseWrapper, mockDataConverter, true)
	if err == nil {
		t.Error("Expected error, got nil")
	}
}

func TestNewInferResult_InvalidHeaderLength(t *testing.T) {
	mockController := gomock.NewController(t)
	defer mockController.Finish()
	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)
	mockResponseWrapper.EXPECT().GetHeader("Inference-Header-Content-Length").Return("invalid")
	mockResponseWrapper.EXPECT().GetHeader("Content-Encoding").Return("")
	mockResponseWrapper.EXPECT().GetBody().Return([]byte("data"), nil)
	mockDataConverter := mocks.NewMockDataConverter(mockController)
	_, err := NewInferResult(mockResponseWrapper, mockDataConverter, true)
	if err == nil {
		t.Error("Expected error, got nil")
	}
}
