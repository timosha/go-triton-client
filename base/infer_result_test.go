package base

import (
	"github.com/Trendyol/go-triton-client/mocks"
	"go.uber.org/mock/gomock"
	"reflect"
	"testing"
)

func TestBaseInferResult_GetOutput_NoOutputs(t *testing.T) {
	result := &BaseInferResult{
		OutputsResponse: InferOutputs{
			Outputs: []*BaseInferOutput{},
		},
	}
	_, err := result.GetOutput("output0")
	if err == nil || err.Error() != "no outputs found in result" {
		t.Errorf("Expected error 'no outputs found in result', got %v", err)
	}
}

func TestBaseInferResult_GetOutput_Found(t *testing.T) {
	mockController := gomock.NewController(t)
	defer mockController.Finish()
	mockOutput := mocks.NewMockInferOutput(mockController)
	mockOutput.EXPECT().GetName().Return("output0")
	result := &BaseInferResult{
		OutputsResponse: InferOutputs{
			Outputs: []*BaseInferOutput{
				{Name: "output0"},
			},
		},
	}
	res, err := result.GetOutput("output0")
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if res.GetName() != mockOutput.GetName() {
		t.Errorf("Expected output name %s, got %s", mockOutput.GetName(), res.GetName())
	}
}

func TestBaseInferResult_GetOutput_NotFound(t *testing.T) {
	result := &BaseInferResult{
		OutputsResponse: InferOutputs{
			Outputs: []*BaseInferOutput{
				{Name: "output1"},
			},
		},
	}
	_, err := result.GetOutput("output0")
	if err == nil || err.Error() != "output output0 not found" {
		t.Errorf("Expected error 'output output0 not found', got %v", err)
	}
}

func TestBaseInferResult_GetShape(t *testing.T) {
	result := &BaseInferResult{
		OutputsResponse: InferOutputs{
			Outputs: []*BaseInferOutput{
				{Name: "output0", Shape: []int64{1, 2}},
			},
		},
	}
	expectedShape := []int64{1, 2}
	res, err := result.GetShape("output0")
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if !reflect.DeepEqual(res, expectedShape) {
		t.Errorf("Expected shape %v, got %v", expectedShape, res)
	}
}

func TestBaseInferResult_GetShape_NotFound(t *testing.T) {
	result := &BaseInferResult{
		OutputsResponse: InferOutputs{
			Outputs: []*BaseInferOutput{
				{Name: "output0", Shape: []int64{1, 2}},
			},
		},
	}
	res, err := result.GetShape("output1")
	if err == nil {
		t.Error("Expected error, got nil", err)
	}
	if res != nil {
		t.Errorf("Expected nil, got %v", res)
	}
}
