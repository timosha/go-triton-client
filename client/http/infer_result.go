package http

import (
	"bytes"
	"compress/gzip"
	"compress/zlib"
	"encoding/json"
	"fmt"
	"github.com/Trendyol/go-triton-client/base"
	"github.com/Trendyol/go-triton-client/converter"
	"io"
	"strconv"
)

// InferResult represents the result of an inference operation using HTTP.
type InferResult struct {
	*base.BaseInferResult
}

// NewInferResult creates a new HTTP InferResult instance.
func NewInferResult(response base.ResponseWrapper, dataConverter converter.DataConverter, verbose bool) (base.InferResult, error) {
	headerLength := response.GetHeader("Inference-Header-Content-Length")

	var decompressedData []byte
	var err error

	contentEncoding := response.GetHeader("Content-Encoding")
	body, err := response.GetBody()
	if err != nil {
		return nil, err
	}

	switch contentEncoding {
	case "gzip":
		reader, err := gzip.NewReader(bytes.NewReader(body))
		if err != nil {
			return nil, err
		}
		decompressedData, err = io.ReadAll(reader)
		if err != nil {
			return nil, err
		}
	case "deflate":
		reader, err := zlib.NewReader(bytes.NewReader(body))
		if err != nil {
			return nil, err
		}
		decompressedData, err = io.ReadAll(reader)
		if err != nil {
			return nil, err
		}
	default:
		decompressedData = body
	}

	if headerLength == "" {
		content := decompressedData
		if verbose {
			fmt.Println(string(content))
		}
		var result base.InferOutputs
		if err := json.Unmarshal(content, &result); err != nil {
			return nil, err
		}

		return &InferResult{
			BaseInferResult: &base.BaseInferResult{
				OutputsResponse: result,
				DataConverter:   dataConverter,
			},
		}, nil
	}

	headerLengthInt, err := strconv.Atoi(headerLength)
	if err != nil {
		return nil, err
	}
	content := decompressedData[:headerLengthInt]
	if verbose {
		fmt.Println(string(content))
	}

	var result base.InferOutputs
	if err := json.Unmarshal(content, &result); err != nil {
		return nil, err
	}

	buffer := decompressedData[headerLengthInt:]
	outputNameToBufferMap := make(map[string]int)
	bufferIndex := 0
	for _, output := range result.Outputs {
		thisDataSize, ok := output.GetParameters()["binary_data_size"].(float64)
		if ok {
			outputNameToBufferMap[output.GetName()] = bufferIndex
			bufferIndex += int(thisDataSize)
		}
	}

	return &InferResult{
		BaseInferResult: &base.BaseInferResult{
			OutputsResponse:       result,
			OutputNameToBufferMap: outputNameToBufferMap,
			Buffer:                buffer,
			DataConverter:         dataConverter,
		},
	}, nil
}

// AsSlice gets the tensor data for the output associated with this object in slice format.
// It checks if the output has associated binary data and deserializes it using the DataConverter.
// The function returns the reshaped slice or an error if the process fails.
func (r *InferResult) AsSlice(name string) (interface{}, error) {
	output, err := r.GetOutput(name)
	if err != nil {
		return nil, err
	}

	hasBinaryData := false
	var slice interface{}
	thisDataSize, ok := output.GetParameters()["binary_data_size"].(float64)
	if ok {
		hasBinaryData = true
		if thisDataSize != 0 {
			startIndex := r.OutputNameToBufferMap[name]
			endIndex := startIndex + int(thisDataSize)
			dataBuffer := r.Buffer[startIndex:endIndex]
			slice, err = r.DataConverter.DeserializeTensor(output.GetDatatype(), dataBuffer)
			if err != nil {
				return nil, err
			}
			return r.DataConverter.ReshapeArray(slice, output.GetShape())
		}
	}

	if !hasBinaryData {
		slice, err = r.DataConverter.ReshapeArray(output.GetData(), output.GetShape())
		if err != nil {
			return nil, err
		}
	}

	return slice, nil
}
