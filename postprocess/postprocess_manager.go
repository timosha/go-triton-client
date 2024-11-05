package postprocess

import (
	"errors"
)

// Converter defines methods for converting slices.
type Converter interface {
	Float64ToFloat32Slice3D(input [][][]float64) [][][]float32
	Float64ToFloat32Slice2D(input [][]float64) [][]float32
}

// Pooling defines methods for performing mean pooling on token embeddings.
type Pooling interface {
	MeanPoolingFloat64Slice3D(tokenEmbeddings [][][]float64, attentionMask [][]int64) ([][]float64, error)
	MeanPoolingFloat32Slice3D(tokenEmbeddings [][][]float32, attentionMask [][]int64) ([][]float32, error)
}

// PostprocessManager aggregates Converter and Pooling interfaces.
type PostprocessManager interface {
	Converter
	Pooling
}

// postprocessManager is the concrete implementation of PostprocessManager.
type postprocessManager struct{}

// NewPostprocessManager creates and returns a new instance of postprocessManager.
func NewPostprocessManager() PostprocessManager {
	return &postprocessManager{}
}

// Float64ToFloat32Slice3D converts a 3D slice of float64 to a 3D slice of float32.
func (pm *postprocessManager) Float64ToFloat32Slice3D(input [][][]float64) [][][]float32 {
	xLen := len(input)
	output := make([][][]float32, xLen)
	for i := range input {
		yLen := len(input[i])
		output[i] = make([][]float32, yLen)
		for j := range input[i] {
			zLen := len(input[i][j])
			output[i][j] = make([]float32, zLen)
			for k := range input[i][j] {
				output[i][j][k] = float32(input[i][j][k])
			}
		}
	}
	return output
}

// Float64ToFloat32Slice2D converts a 2D slice of float64 to a 2D slice of float32.
func (pm *postprocessManager) Float64ToFloat32Slice2D(input [][]float64) [][]float32 {
	xLen := len(input)
	output := make([][]float32, xLen)
	for i := range input {
		yLen := len(input[i])
		output[i] = make([]float32, yLen)
		for j := range input[i] {
			output[i][j] = float32(input[i][j])
		}
	}
	return output
}

// MeanPoolingFloat64Slice3D performs mean pooling on float64 token embeddings.
func (pm *postprocessManager) MeanPoolingFloat64Slice3D(tokenEmbeddings [][][]float64, attentionMask [][]int64) ([][]float64, error) {
	return performMeanPooling[float64](tokenEmbeddings, attentionMask)
}

// MeanPoolingFloat32Slice3D performs mean pooling on float32 token embeddings.
func (pm *postprocessManager) MeanPoolingFloat32Slice3D(tokenEmbeddings [][][]float32, attentionMask [][]int64) ([][]float32, error) {
	return performMeanPooling[float32](tokenEmbeddings, attentionMask)
}

// performMeanPooling is a generic function to perform mean pooling on token embeddings.
func performMeanPooling[T float32 | float64](tokenEmbeddings [][][]T, attentionMask [][]int64) ([][]T, error) {
	if len(tokenEmbeddings) == 0 || len(attentionMask) == 0 {
		return nil, errors.New("empty input slices")
	}

	batchSize := len(tokenEmbeddings)
	sequenceLength := len(tokenEmbeddings[0])
	embeddingSize := len(tokenEmbeddings[0][0])

	if err := validateDimensions(tokenEmbeddings, attentionMask, batchSize, sequenceLength); err != nil {
		return nil, err
	}

	sumEmbeddings := make([][]T, batchSize)
	sumMask := make([]T, batchSize)
	for i := 0; i < batchSize; i++ {
		sumEmbeddings[i] = make([]T, embeddingSize)
	}

	for i := 0; i < batchSize; i++ {
		for j := 0; j < sequenceLength; j++ {
			maskValue := T(attentionMask[i][j])
			sumMask[i] += maskValue
			for k := 0; k < embeddingSize; k++ {
				sumEmbeddings[i][k] += tokenEmbeddings[i][j][k] * maskValue
			}
		}
	}

	epsilon := T(1e-9)
	for i := 0; i < batchSize; i++ {
		if sumMask[i] < epsilon {
			sumMask[i] = epsilon
		}
	}

	meanEmbeddings := make([][]T, batchSize)
	for i := 0; i < batchSize; i++ {
		meanEmbeddings[i] = make([]T, embeddingSize)
		for k := 0; k < embeddingSize; k++ {
			meanEmbeddings[i][k] = sumEmbeddings[i][k] / sumMask[i]
		}
	}

	return meanEmbeddings, nil
}

// validateDimensions checks if the dimensions of tokenEmbeddings and attentionMask are consistent.
func validateDimensions[T float32 | float64](tokenEmbeddings [][][]T, attentionMask [][]int64, batchSize, sequenceLength int) error {
	if len(attentionMask) != batchSize {
		return errors.New("mismatched batch size between tokenEmbeddings and attentionMask")
	}
	for i := 0; i < batchSize; i++ {
		if len(tokenEmbeddings[i]) != sequenceLength {
			return errors.New("mismatched sequence length in tokenEmbeddings")
		}
		if len(attentionMask[i]) != sequenceLength {
			return errors.New("mismatched sequence length in attentionMask")
		}
		for j := 0; j < sequenceLength; j++ {
			if len(tokenEmbeddings[i][j]) == 0 {
				return errors.New("embedding vector cannot be empty")
			}
		}
	}
	return nil
}
