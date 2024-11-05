package postprocess

import (
	"reflect"
	"testing"
)

func TestFloat64ToFloat32Slice3D(t *testing.T) {
	pm := NewPostprocessManager()
	input := [][][]float64{
		{
			{1.1, 2.2},
			{3.3, 4.4},
		},
		{
			{5.5, 6.6},
			{7.7, 8.8},
		},
	}
	expected := [][][]float32{
		{
			{1.1, 2.2},
			{3.3, 4.4},
		},
		{
			{5.5, 6.6},
			{7.7, 8.8},
		},
	}
	result := pm.Float64ToFloat32Slice3D(input)
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("Float64ToFloat32Slice3D failed. Expected %v, got %v", expected, result)
	}
}

func TestFloat64ToFloat32Slice2D(t *testing.T) {
	pm := NewPostprocessManager()
	input := [][]float64{
		{1.1, 2.2},
		{3.3, 4.4},
	}
	expected := [][]float32{
		{1.1, 2.2},
		{3.3, 4.4},
	}
	result := pm.Float64ToFloat32Slice2D(input)
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("Float64ToFloat32Slice2D failed. Expected %v, got %v", expected, result)
	}
}

func TestMeanPoolingFloat64Slice3D(t *testing.T) {
	pm := NewPostprocessManager()
	tokenEmbeddings := [][][]float64{
		{
			{1.0, 2.0},
			{3.0, 4.0},
		},
		{
			{5.0, 6.0},
			{7.0, 8.0},
		},
	}
	attentionMask := [][]int64{
		{1, 1},
		{0, 1},
	}
	expected := [][]float64{
		{2.0, 3.0},
		{7.0, 8.0},
	}
	result, err := pm.MeanPoolingFloat64Slice3D(tokenEmbeddings, attentionMask)
	if err != nil {
		t.Errorf("MeanPoolingFloat64Slice3D returned an error: %v", err)
	}
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("MeanPoolingFloat64Slice3D failed. Expected %v, got %v", expected, result)
	}
}

func TestMeanPoolingFloat32Slice3D(t *testing.T) {
	pm := NewPostprocessManager()
	tokenEmbeddings := [][][]float32{
		{
			{1.0, 2.0},
			{3.0, 4.0},
		},
		{
			{5.0, 6.0},
			{7.0, 8.0},
		},
	}
	attentionMask := [][]int64{
		{1, 1},
		{0, 1},
	}
	expected := [][]float32{
		{2.0, 3.0},
		{7.0, 8.0},
	}
	result, err := pm.MeanPoolingFloat32Slice3D(tokenEmbeddings, attentionMask)
	if err != nil {
		t.Errorf("MeanPoolingFloat32Slice3D returned an error: %v", err)
	}
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("MeanPoolingFloat32Slice3D failed. Expected %v, got %v", expected, result)
	}
}

func TestMeanPoolingEmptyInput(t *testing.T) {
	pm := NewPostprocessManager()
	_, err := pm.MeanPoolingFloat64Slice3D([][][]float64{}, [][]int64{})
	if err == nil {
		t.Errorf("Expected error for empty input, got nil")
	}
	_, err = pm.MeanPoolingFloat32Slice3D([][][]float32{}, [][]int64{})
	if err == nil {
		t.Errorf("Expected error for empty input, got nil")
	}
}

func TestMeanPoolingMismatchedDimensions(t *testing.T) {
	pm := NewPostprocessManager()
	tokenEmbeddings := [][][]float64{
		{
			{1.0, 2.0},
		},
	}
	attentionMask := [][]int64{
		{1, 1},
	}
	_, err := pm.MeanPoolingFloat64Slice3D(tokenEmbeddings, attentionMask)
	if err == nil {
		t.Errorf("Expected error for mismatched dimensions, got nil")
	}
}

func TestMeanPoolingZeroDivision(t *testing.T) {
	pm := NewPostprocessManager()
	tokenEmbeddings := [][][]float64{
		{
			{1.0, 2.0},
			{3.0, 4.0},
		},
	}
	attentionMask := [][]int64{
		{0, 0},
	}
	result, err := pm.MeanPoolingFloat64Slice3D(tokenEmbeddings, attentionMask)
	if err != nil {
		t.Errorf("MeanPoolingFloat64Slice3D returned an error: %v", err)
	}
	if result[0][0] != 0 || result[0][1] != 0 {
		t.Errorf("Expected [[0, 0]], got %v", result)
	}
}

func TestFloat64ToFloat32Slice3DEmptyInput(t *testing.T) {
	pm := NewPostprocessManager()
	input := [][][]float64{}
	expected := [][][]float32{}
	result := pm.Float64ToFloat32Slice3D(input)
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("Expected empty slice, got %v", result)
	}
}

func TestFloat64ToFloat32Slice2DEmptyInput(t *testing.T) {
	pm := NewPostprocessManager()
	input := [][]float64{}
	expected := [][]float32{}
	result := pm.Float64ToFloat32Slice2D(input)
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("Expected empty slice, got %v", result)
	}
}

func TestMeanPoolingNegativeValues(t *testing.T) {
	pm := NewPostprocessManager()
	tokenEmbeddings := [][][]float64{
		{
			{-1.0, -2.0},
			{-3.0, -4.0},
		},
	}
	attentionMask := [][]int64{
		{1, 1},
	}
	expected := [][]float64{
		{-2.0, -3.0},
	}
	result, err := pm.MeanPoolingFloat64Slice3D(tokenEmbeddings, attentionMask)
	if err != nil {
		t.Errorf("MeanPoolingFloat64Slice3D returned an error: %v", err)
	}
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("Expected %v, got %v", expected, result)
	}
}

func TestValidateDimensions(t *testing.T) {
	tokenEmbeddings := [][][]float64{
		{
			{1.0},
		},
	}
	attentionMask := [][]int64{
		{1},
	}
	err := validateDimensions(tokenEmbeddings, attentionMask, 1, 1)
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	tokenEmbeddings[0][0] = []float64{}
	err = validateDimensions(tokenEmbeddings, attentionMask, 1, 1)
	if err == nil {
		t.Errorf("Expected error for empty embedding vector, got nil")
	}
}

func TestNewPostprocessManager(t *testing.T) {
	pm := NewPostprocessManager()
	if pm == nil {
		t.Errorf("NewPostprocessManager returned nil")
	}
}

func TestValidateDimensionsMismatchedBatchSize(t *testing.T) {
	tokenEmbeddings := [][][]float64{
		{
			{1.0, 2.0},
		},
		{
			{3.0, 4.0},
		},
	}
	attentionMask := [][]int64{
		{1, 1},
	}
	err := validateDimensions(tokenEmbeddings, attentionMask, len(tokenEmbeddings), len(tokenEmbeddings[0]))
	if err == nil || err.Error() != "mismatched batch size between tokenEmbeddings and attentionMask" {
		t.Errorf("Expected mismatched batch size error, got %v", err)
	}
}

func TestValidateDimensionsMismatchedSequenceLengthTokenEmbeddings(t *testing.T) {
	tokenEmbeddings := [][][]float64{
		{
			{1.0, 2.0},
			{3.0, 4.0},
		},
		{
			{5.0, 6.0},
		},
	}
	attentionMask := [][]int64{
		{1, 1},
		{1, 1},
	}
	err := validateDimensions(tokenEmbeddings, attentionMask, len(tokenEmbeddings), len(tokenEmbeddings[0]))
	if err == nil || err.Error() != "mismatched sequence length in tokenEmbeddings" {
		t.Errorf("Expected mismatched sequence length in tokenEmbeddings error, got %v", err)
	}
}

func TestValidateDimensionsMismatchedSequenceLengthAttentionMask(t *testing.T) {
	tokenEmbeddings := [][][]float64{
		{
			{1.0, 2.0},
			{3.0, 4.0},
		},
		{
			{5.0, 6.0},
			{7.0, 8.0},
		},
	}
	attentionMask := [][]int64{
		{1, 1},
		{1},
	}
	err := validateDimensions(tokenEmbeddings, attentionMask, len(tokenEmbeddings), len(tokenEmbeddings[0]))
	if err == nil || err.Error() != "mismatched sequence length in attentionMask" {
		t.Errorf("Expected mismatched sequence length in attentionMask error, got %v", err)
	}
}

func TestValidateDimensionsEmptyEmbeddingVector(t *testing.T) {
	tokenEmbeddings := [][][]float64{
		{
			{1.0, 2.0},
			{},
		},
	}
	attentionMask := [][]int64{
		{1, 1},
	}
	err := validateDimensions(tokenEmbeddings, attentionMask, len(tokenEmbeddings), len(tokenEmbeddings[0]))
	if err == nil || err.Error() != "embedding vector cannot be empty" {
		t.Errorf("Expected embedding vector cannot be empty error, got %v", err)
	}
}
