package converter

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"github.com/x448/float16"
	"io"
	"math"
	"reflect"
)

// DataConverter is an interface that defines methods for converting binary data to various numerical types,
// deserializing tensors based on their data type, and reshaping arrays to a specified shape.
type DataConverter interface {
	SerializeTensor(inputTensor interface{}) ([]byte, error)
	DeserializeTensor(datatype string, dataBuffer []byte) (interface{}, error)
	FlattenData(inputTensor interface{}) []interface{}
	ReshapeArray(data interface{}, shape []int) ([]interface{}, error)
	ConvertByteSliceToInt32Slice(data []byte) []int32
	ConvertByteSliceToInt64Slice(data []byte) []int64
	ConvertByteSliceToFloat32Slice(data []byte) []float32
	ConvertByteSliceToFloat64Slice(data []byte) []float64
	ConvertInterfaceSliceToFloat32SliceAsInterface(data []interface{}) []interface{}
	ConvertInterfaceSliceToFloat64SliceAsInterface(data []interface{}) []interface{}
	ConvertInterfaceSliceToInt32SliceAsInterface(data []interface{}) []interface{}
	ConvertInterfaceSliceToInt64SliceAsInterface(data []interface{}) []interface{}
	ConvertInterfaceSliceToUint32SliceAsInterface(data []interface{}) []interface{}
	ConvertInterfaceSliceToUint64SliceAsInterface(data []interface{}) []interface{}
	ConvertInterfaceSliceToBoolSliceAsInterface(data []interface{}) []interface{}
	ConvertInterfaceSliceToBytesSliceAsInterface(data []interface{}) ([]interface{}, error)
}

type dataConverter struct{}

// NewDataConverter creates and returns a new instance of dataConverter.
func NewDataConverter() DataConverter {
	return &dataConverter{}
}

// SerializeTensor serializes a tensor into a byte slice based on its datatype.
func (dc *dataConverter) SerializeTensor(inputTensor interface{}) ([]byte, error) {
	var buffer bytes.Buffer
	err := dc.serializeTensorToWriter(inputTensor, &buffer)
	if err != nil {
		return nil, err
	}
	return buffer.Bytes(), nil
}

func (dc *dataConverter) serializeTensorToWriter(inputTensor interface{}, w io.Writer) error {
	switch tensor := inputTensor.(type) {
	case []int:
		for _, v := range tensor {
			if err := binary.Write(w, binary.LittleEndian, int64(v)); err != nil {
				return err
			}
		}
	case []int32:
		for _, v := range tensor {
			if err := binary.Write(w, binary.LittleEndian, v); err != nil {
				return err
			}
		}
	case []int64:
		for _, v := range tensor {
			if err := binary.Write(w, binary.LittleEndian, v); err != nil {
				return err
			}
		}
	case []uint16:
		for _, v := range tensor {
			if err := binary.Write(w, binary.LittleEndian, v); err != nil {
				return err
			}
		}
	case []uint32:
		for _, v := range tensor {
			if err := binary.Write(w, binary.LittleEndian, v); err != nil {
				return err
			}
		}
	case []uint64:
		for _, v := range tensor {
			if err := binary.Write(w, binary.LittleEndian, v); err != nil {
				return err
			}
		}
	case []float32:
		for _, v := range tensor {
			if err := binary.Write(w, binary.LittleEndian, v); err != nil {
				return err
			}
		}
	case []float64:
		for _, v := range tensor {
			if err := binary.Write(w, binary.LittleEndian, v); err != nil {
				return err
			}
		}
	case []bool:
		for _, v := range tensor {
			var boolVal byte
			if v {
				boolVal = 1
			} else {
				boolVal = 0
			}
			if err := binary.Write(w, binary.LittleEndian, boolVal); err != nil {
				return err
			}
		}
	case []byte:
		if _, err := w.Write(tensor); err != nil {
			return err
		}
	case []string:
		for _, str := range tensor {
			strBytes := []byte(str)
			strLen := int32(len(strBytes))
			if err := binary.Write(w, binary.LittleEndian, strLen); err != nil {
				return err
			}
			if _, err := w.Write(strBytes); err != nil {
				return err
			}
		}
	default:
		return errors.New("unsupported tensor datatype")
	}
	return nil
}

// DeserializeTensor deserializes a byte slice into a specific tensor type based on the provided data type string.
// It returns the deserialized tensor as an interface{} or an error if the data type is not supported.
func (dc *dataConverter) DeserializeTensor(datatype string, dataBuffer []byte) (interface{}, error) {
	switch datatype {
	case "BOOL":
		return dc.deserializeBoolTensor(dataBuffer), nil
	case "INT8":
		return dc.deserializeInt8Tensor(dataBuffer), nil
	case "INT16":
		return dc.deserializeInt16Tensor(dataBuffer), nil
	case "INT32":
		return dc.deserializeInt32Tensor(dataBuffer), nil
	case "INT64":
		return dc.deserializeInt64Tensor(dataBuffer), nil
	case "UINT8":
		return dc.deserializeUint8Tensor(dataBuffer), nil
	case "UINT16":
		return dc.deserializeUint16Tensor(dataBuffer), nil
	case "UINT32":
		return dc.deserializeUint32Tensor(dataBuffer), nil
	case "UINT64":
		return dc.deserializeUint64Tensor(dataBuffer), nil
	case "FP16":
		return dc.deserializeFloat16Tensor(dataBuffer), nil
	case "FP32":
		return dc.deserializeFloat32Tensor(dataBuffer), nil
	case "FP64":
		return dc.deserializeFloat64Tensor(dataBuffer), nil
	case "BYTES":
		return dc.deserializeBytesTensor(dataBuffer)
	case "BF16":
		return dc.deserializeBF16Tensor(dataBuffer)
	default:
		return nil, fmt.Errorf("unsupported datatype: %s", datatype)
	}
}

// FlattenData converts a multi-dimensional tensor into a 1D slice of interfaces.
func (dc *dataConverter) FlattenData(inputTensor interface{}) []interface{} {
	switch tensor := inputTensor.(type) {
	case []int:
		result := make([]interface{}, len(tensor))
		for i, v := range tensor {
			result[i] = v
		}
		return result
	case []int32:
		result := make([]interface{}, len(tensor))
		for i, v := range tensor {
			result[i] = v
		}
		return result
	case []int64:
		result := make([]interface{}, len(tensor))
		for i, v := range tensor {
			result[i] = v
		}
		return result
	case []uint16:
		result := make([]interface{}, len(tensor))
		for i, v := range tensor {
			result[i] = v
		}
		return result
	case []uint32:
		result := make([]interface{}, len(tensor))
		for i, v := range tensor {
			result[i] = v
		}
		return result
	case []uint64:
		result := make([]interface{}, len(tensor))
		for i, v := range tensor {
			result[i] = v
		}
		return result
	case []float32:
		result := make([]interface{}, len(tensor))
		for i, v := range tensor {
			result[i] = v
		}
		return result
	case []float64:
		result := make([]interface{}, len(tensor))
		for i, v := range tensor {
			result[i] = v
		}
		return result
	case []byte:
		result := make([]interface{}, len(tensor))
		for i, v := range tensor {
			result[i] = v
		}
		return result
	case []bool:
		result := make([]interface{}, len(tensor))
		for i, v := range tensor {
			result[i] = v
		}
		return result
	case []string:
		result := make([]interface{}, len(tensor))
		for i, v := range tensor {
			result[i] = v
		}
		return result
	default:
		return nil
	}
}

func (dc *dataConverter) ReshapeArray(data interface{}, shape []int) ([]interface{}, error) {
	if len(shape) == 0 {
		return nil, fmt.Errorf("shape cannot be empty")
	}

	for _, dim := range shape {
		if dim <= 0 {
			return nil, fmt.Errorf("invalid shape: dimensions must be positive integers")
		}
	}

	rv := reflect.ValueOf(data)

	return dc.reshapeRecursively(rv, shape)
}

// reshapeRecursively is a helper function that recursively reshapes a flat array of data into a multi-dimensional array.
func (dc *dataConverter) reshapeRecursively(rv reflect.Value, shape []int) ([]interface{}, error) {
	if len(shape) == 0 {
		return nil, fmt.Errorf("shape cannot be empty")
	}

	if len(shape) == 1 {
		if rv.Len() != shape[0] {
			return nil, fmt.Errorf("data length does not match shape")
		}
		subArrays := make([]interface{}, rv.Len())
		for i := 0; i < rv.Len(); i++ {
			subArrays[i] = rv.Index(i).Interface()
		}
		return subArrays, nil
	}

	if rv.Len()%shape[0] != 0 {
		return nil, fmt.Errorf("data cannot be evenly divided according to the shape")
	}

	subSize := rv.Len() / shape[0]
	subArrays := make([]interface{}, shape[0])

	for i := 0; i < shape[0]; i++ {
		subSlice := rv.Slice(i*subSize, (i+1)*subSize)

		subArray, err := dc.reshapeRecursively(subSlice, shape[1:])
		if err != nil {
			return nil, err
		}
		subArrays[i] = subArray
	}
	return subArrays, nil
}

// ConvertByteSliceToInt32Slice converts a byte slice to a slice of int32
func (dc *dataConverter) ConvertByteSliceToInt32Slice(data []byte) []int32 {
	var int32Data []int32
	for _, b := range data {
		int32Data = append(int32Data, int32(b))
	}
	return int32Data
}

// ConvertByteSliceToInt64Slice converts a byte slice to a slice of int64
func (dc *dataConverter) ConvertByteSliceToInt64Slice(data []byte) []int64 {
	var int64Data []int64
	for i := 0; i < len(data); i += 8 {
		int64Data = append(int64Data, int64(binary.LittleEndian.Uint64(data[i:i+8])))
	}
	return int64Data
}

// ConvertByteSliceToFloat32Slice converts a byte slice to a slice of float32
func (dc *dataConverter) ConvertByteSliceToFloat32Slice(data []byte) []float32 {
	var float32Data []float32
	for i := 0; i < len(data); i += 4 {
		float32Data = append(float32Data, math.Float32frombits(binary.LittleEndian.Uint32(data[i:i+4])))
	}
	return float32Data
}

// ConvertByteSliceToFloat64Slice converts a byte slice to a slice of float64
func (dc *dataConverter) ConvertByteSliceToFloat64Slice(data []byte) []float64 {
	var float64Data []float64
	for i := 0; i < len(data); i += 8 {
		float64Data = append(float64Data, math.Float64frombits(binary.LittleEndian.Uint64(data[i:i+8])))
	}
	return float64Data
}

// ConvertInterfaceSliceToFloat32SliceAsInterface converts a interface slice to a slice of float32 as interface
func (dc *dataConverter) ConvertInterfaceSliceToFloat32SliceAsInterface(data []interface{}) []interface{} {
	convertedData := make([]interface{}, len(data))
	for i, v := range data {
		convertedData[i] = float32(v.(float64))
	}
	return convertedData
}

// ConvertInterfaceSliceToFloat64SliceAsInterface converts a interface slice to a slice of float64 as interface
func (dc *dataConverter) ConvertInterfaceSliceToFloat64SliceAsInterface(data []interface{}) []interface{} {
	convertedData := make([]interface{}, len(data))
	for i, v := range data {
		convertedData[i] = v.(float64)
	}
	return convertedData
}

// ConvertInterfaceSliceToInt32SliceAsInterface converts a interface slice to a slice of int32 as interface
func (dc *dataConverter) ConvertInterfaceSliceToInt32SliceAsInterface(data []interface{}) []interface{} {
	convertedData := make([]interface{}, len(data))
	for i, v := range data {
		convertedData[i] = int32(v.(float64))
	}
	return convertedData
}

// ConvertInterfaceSliceToInt64SliceAsInterface converts a interface slice to a slice of int64 as interface
func (dc *dataConverter) ConvertInterfaceSliceToInt64SliceAsInterface(data []interface{}) []interface{} {
	convertedData := make([]interface{}, len(data))
	for i, v := range data {
		convertedData[i] = int64(v.(float64))
	}
	return convertedData
}

// ConvertInterfaceSliceToUint32SliceAsInterface converts a interface slice to a slice of uint32 as interface
func (dc *dataConverter) ConvertInterfaceSliceToUint32SliceAsInterface(data []interface{}) []interface{} {
	convertedData := make([]interface{}, len(data))
	for i, v := range data {
		convertedData[i] = uint32(v.(float64))
	}
	return convertedData
}

// ConvertInterfaceSliceToUint64SliceAsInterface converts a interface slice to a slice of uint64 as interface
func (dc *dataConverter) ConvertInterfaceSliceToUint64SliceAsInterface(data []interface{}) []interface{} {
	convertedData := make([]interface{}, len(data))
	for i, v := range data {
		convertedData[i] = uint64(v.(float64))
	}
	return convertedData
}

// ConvertInterfaceSliceToBoolSliceAsInterface converts a interface slice to a slice of bool as interface
func (dc *dataConverter) ConvertInterfaceSliceToBoolSliceAsInterface(data []interface{}) []interface{} {
	convertedData := make([]interface{}, len(data))
	for i, v := range data {
		convertedData[i] = v.(bool)
	}
	return convertedData
}

// ConvertInterfaceSliceToBytesSliceAsInterface converts a interface slice to a slice of bytes as interface
func (dc *dataConverter) ConvertInterfaceSliceToBytesSliceAsInterface(data []interface{}) ([]interface{}, error) {
	convertedData := make([]interface{}, len(data))
	for i, v := range data {
		str, ok := v.(string)
		if !ok {
			return nil, fmt.Errorf("expected BYTES datatype, got %T", v)
		}
		convertedData[i] = []byte(str)
	}
	return convertedData, nil
}

// deserializeInt8Tensor converts a byte slice to a slice of int8 values.
func (dc *dataConverter) deserializeInt8Tensor(dataBuffer []byte) []int8 {
	int8Array := make([]int8, len(dataBuffer))
	for i, v := range dataBuffer {
		int8Array[i] = int8(v)
	}
	return int8Array
}

// deserializeInt16Tensor converts a byte slice to a slice of int16 values by interpreting each 2-byte chunk as an int16.
func (dc *dataConverter) deserializeInt16Tensor(dataBuffer []byte) []int16 {
	int16Array := make([]int16, len(dataBuffer)/2)
	for i := 0; i < len(dataBuffer); i += 2 {
		int16Array[i/2] = int16(binary.LittleEndian.Uint16(dataBuffer[i:]))
	}
	return int16Array
}

// deserializeInt32Tensor converts a byte slice to a slice of int32 values by interpreting each 4-byte chunk as an int32.
func (dc *dataConverter) deserializeInt32Tensor(dataBuffer []byte) []int32 {
	int32Array := make([]int32, len(dataBuffer)/4)
	for i := 0; i < len(dataBuffer); i += 4 {
		int32Array[i/4] = int32(binary.LittleEndian.Uint32(dataBuffer[i:]))
	}
	return int32Array
}

// deserializeInt64Tensor converts a byte slice to a slice of int64 values by interpreting each 8-byte chunk as an int64.
func (dc *dataConverter) deserializeInt64Tensor(dataBuffer []byte) []int64 {
	int64Array := make([]int64, len(dataBuffer)/8)
	for i := 0; i < len(dataBuffer); i += 8 {
		int64Array[i/8] = int64(binary.LittleEndian.Uint64(dataBuffer[i:]))
	}
	return int64Array
}

// deserializeUint8Tensor returns the byte slice directly as a slice of uint8 values.
func (dc *dataConverter) deserializeUint8Tensor(dataBuffer []byte) []uint8 {
	return dataBuffer
}

// deserializeUint16Tensor converts a byte slice to a slice of uint16 values by interpreting each 2-byte chunk as a uint16.
func (dc *dataConverter) deserializeUint16Tensor(dataBuffer []byte) []uint16 {
	uint16Array := make([]uint16, len(dataBuffer)/2)
	for i := 0; i < len(dataBuffer); i += 2 {
		uint16Array[i/2] = binary.LittleEndian.Uint16(dataBuffer[i:])
	}
	return uint16Array
}

// deserializeUint32Tensor converts a byte slice to a slice of uint32 values by interpreting each 4-byte chunk as a uint32.
func (dc *dataConverter) deserializeUint32Tensor(dataBuffer []byte) []uint32 {
	uint32Array := make([]uint32, len(dataBuffer)/4)
	for i := 0; i < len(dataBuffer); i += 4 {
		uint32Array[i/4] = binary.LittleEndian.Uint32(dataBuffer[i:])
	}
	return uint32Array
}

// deserializeUint64Tensor converts a byte slice to a slice of uint64 values by interpreting each 8-byte chunk as a uint64.
func (dc *dataConverter) deserializeUint64Tensor(dataBuffer []byte) []uint64 {
	uint64Array := make([]uint64, len(dataBuffer)/8)
	for i := 0; i < len(dataBuffer); i += 8 {
		uint64Array[i/8] = binary.LittleEndian.Uint64(dataBuffer[i:])
	}
	return uint64Array
}

// deserializeBoolTensor converts a byte slice to a slice of boolean values where each byte is treated as a boolean.
func (dc *dataConverter) deserializeBoolTensor(dataBuffer []byte) []bool {
	boolArray := make([]bool, len(dataBuffer))
	for i, v := range dataBuffer {
		boolArray[i] = v != 0
	}
	return boolArray
}

// deserializeBytesTensor deserializes a byte slice into a slice of strings. The byte slice is expected to be encoded
// with the length of each string as a 4-byte little-endian integer followed by the string bytes.
func (dc *dataConverter) deserializeBytesTensor(encodedTensor []byte) ([]string, error) {
	var strings []string
	offset := 0

	for offset < len(encodedTensor) {
		if offset+4 > len(encodedTensor) {
			return nil, fmt.Errorf("unexpected end of encoded tensor")
		}

		length := binary.LittleEndian.Uint32(encodedTensor[offset : offset+4])
		offset += 4

		if offset+int(length) > len(encodedTensor) {
			return nil, fmt.Errorf("unexpected end of encoded tensor")
		}

		str := string(encodedTensor[offset : offset+int(length)])
		offset += int(length)

		strings = append(strings, str)
	}

	return strings, nil
}

// deserializeBF16Tensor converts a byte slice to a slice of float32 values by treating each 2-byte chunk as a bfloat16 value.
func (dc *dataConverter) deserializeBF16Tensor(encodedTensor []byte) ([]float32, error) {
	var floats []float32
	for i := 0; i < len(encodedTensor); i += 2 {
		bits := binary.LittleEndian.Uint16(encodedTensor[i : i+2])
		float32Bits := uint32(bits) << 16
		floats = append(floats, math.Float32frombits(float32Bits))
	}
	return floats, nil
}

// deserializeFloat16Tensor converts a byte slice to a slice of float64 values by treating each 2-byte chunk as a float16 value.
func (dc *dataConverter) deserializeFloat16Tensor(encodedTensor []byte) []float64 {
	var floats []float64
	for i := 0; i < len(encodedTensor); i += 2 {
		uint16Value := binary.LittleEndian.Uint16(encodedTensor[i : i+2])
		float16Value := float16.Frombits(uint16Value)
		floats = append(floats, float64(float16Value.Float32()))
	}
	return floats
}

// deserializeFloat32Tensor converts a byte slice to a slice of float32 values by treating each 4-byte chunk as a float32 value.
func (dc *dataConverter) deserializeFloat32Tensor(encodedTensor []byte) []float32 {
	var floats []float32
	for i := 0; i < len(encodedTensor); i += 4 {
		float32Value := math.Float32frombits(binary.LittleEndian.Uint32(encodedTensor[i : i+4]))
		floats = append(floats, float32Value)
	}
	return floats
}

// deserializeFloat64Tensor converts a byte slice to a slice of float64 values by treating each 8-byte chunk as a float64 value.
func (dc *dataConverter) deserializeFloat64Tensor(encodedTensor []byte) []float64 {
	var floats []float64
	for i := 0; i < len(encodedTensor); i += 8 {
		float64Value := math.Float64frombits(binary.LittleEndian.Uint64(encodedTensor[i : i+8]))
		floats = append(floats, float64Value)
	}
	return floats
}
