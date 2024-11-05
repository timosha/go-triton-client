package converter

import (
	"bytes"
	"encoding/binary"
	"errors"
	"github.com/x448/float16"
	"io"
	"math"
	"reflect"
	"strings"
	"testing"
)

type FaultyWriter struct {
	failOnWrite bool
}

func (fw *FaultyWriter) Write(p []byte) (int, error) {
	if fw.failOnWrite {
		return 0, errors.New("write error")
	}
	return len(p), nil
}

type FaultyStringWriter struct {
	underlyingWriter io.Writer
	failOnData       []byte
	failed           bool
}

func (fw *FaultyStringWriter) Write(p []byte) (int, error) {
	if !fw.failed && bytes.Equal(p, fw.failOnData) {
		fw.failed = true
		return 0, errors.New("write error")
	}
	return fw.underlyingWriter.Write(p)
}

func serializeInt64(data []int64) []byte {
	var buf bytes.Buffer
	for _, v := range data {
		binary.Write(&buf, binary.LittleEndian, v)
	}
	return buf.Bytes()
}

func serializeInt32(data []int32) []byte {
	var buf bytes.Buffer
	for _, v := range data {
		binary.Write(&buf, binary.LittleEndian, v)
	}
	return buf.Bytes()
}

func serializeUint16(data []uint16) []byte {
	var buf bytes.Buffer
	for _, v := range data {
		binary.Write(&buf, binary.LittleEndian, v)
	}
	return buf.Bytes()
}

func serializeUint32(data []uint32) []byte {
	var buf bytes.Buffer
	for _, v := range data {
		binary.Write(&buf, binary.LittleEndian, v)
	}
	return buf.Bytes()
}

func serializeUint64(data []uint64) []byte {
	var buf bytes.Buffer
	for _, v := range data {
		binary.Write(&buf, binary.LittleEndian, v)
	}
	return buf.Bytes()
}

func serializeFloat32(data []float32) []byte {
	var buf bytes.Buffer
	for _, v := range data {
		binary.Write(&buf, binary.LittleEndian, v)
	}
	return buf.Bytes()
}

func serializeFloat64(data []float64) []byte {
	var buf bytes.Buffer
	for _, v := range data {
		binary.Write(&buf, binary.LittleEndian, v)
	}
	return buf.Bytes()
}

func serializeStrings(data []string) []byte {
	var buf bytes.Buffer
	for _, str := range data {
		strBytes := []byte(str)
		strLen := int32(len(strBytes))
		binary.Write(&buf, binary.LittleEndian, strLen)
		buf.Write(strBytes)
	}
	return buf.Bytes()
}

func serializeFloat16(data []float64) []byte {
	var buf bytes.Buffer
	for _, v := range data {
		f16 := float16.Fromfloat32(float32(v))
		binary.Write(&buf, binary.LittleEndian, f16.Bits())
	}
	return buf.Bytes()
}

func serializeBF16(data []float32) []byte {
	var buf bytes.Buffer
	for _, v := range data {
		bits := math.Float32bits(v)
		bf16 := uint16(bits >> 16)
		binary.Write(&buf, binary.LittleEndian, bf16)
	}
	return buf.Bytes()
}

func TestNewDataConverter(t *testing.T) {
	dc := NewDataConverter()
	if dc == nil {
		t.Errorf("Expected DataConverter instance, got nil")
	}
}

func TestSerializeTensor(t *testing.T) {
	dc := NewDataConverter()
	testCases := []struct {
		input     interface{}
		expected  []byte
		expectErr bool
	}{
		{[]int{1, 2, 3}, serializeInt64([]int64{1, 2, 3}), false},
		{[]int32{1, 2, 3}, serializeInt32([]int32{1, 2, 3}), false},
		{[]int64{1, 2, 3}, serializeInt64([]int64{1, 2, 3}), false},
		{[]uint16{1, 2, 3}, serializeUint16([]uint16{1, 2, 3}), false},
		{[]uint32{1, 2, 3}, serializeUint32([]uint32{1, 2, 3}), false},
		{[]uint64{1, 2, 3}, serializeUint64([]uint64{1, 2, 3}), false},
		{[]float32{1.0, 2.0, 3.0}, serializeFloat32([]float32{1.0, 2.0, 3.0}), false},
		{[]float64{1.0, 2.0, 3.0}, serializeFloat64([]float64{1.0, 2.0, 3.0}), false},
		{[]bool{true, false}, []byte{1, 0}, false},
		{[]byte{0x01, 0x02}, []byte{0x01, 0x02}, false},
		{[]string{"hello", "world"}, serializeStrings([]string{"hello", "world"}), false},
		{[]struct{}{}, nil, true},
	}
	for _, tc := range testCases {
		result, err := dc.SerializeTensor(tc.input)
		if (err != nil) != tc.expectErr {
			t.Errorf("SerializeTensor(%v) unexpected error status: %v", tc.input, err)
		}
		if !tc.expectErr && !bytes.Equal(result, tc.expected) {
			t.Errorf("SerializeTensor(%v) = %v; want %v", tc.input, result, tc.expected)
		}
	}
}

func TestDeserializeTensor(t *testing.T) {
	dc := NewDataConverter()
	testCases := []struct {
		datatype   string
		dataBuffer []byte
		expected   interface{}
		expectErr  bool
	}{
		{"BOOL", []byte{1, 0}, []bool{true, false}, false},
		{"INT8", []byte{0x01, 0x02}, []int8{1, 2}, false},
		{"INT16", serializeUint16([]uint16{1, 2}), []int16{1, 2}, false},
		{"INT32", serializeInt32([]int32{1, 2}), []int32{1, 2}, false},
		{"INT64", serializeInt64([]int64{1, 2}), []int64{1, 2}, false},
		{"UINT8", []byte{0x01, 0x02}, []uint8{1, 2}, false},
		{"UINT16", serializeUint16([]uint16{1, 2}), []uint16{1, 2}, false},
		{"UINT32", serializeUint32([]uint32{1, 2}), []uint32{1, 2}, false},
		{"UINT64", serializeUint64([]uint64{1, 2}), []uint64{1, 2}, false},
		{"FP16", serializeFloat16([]float64{1.0, 2.0}), []float64{1.0, 2.0}, false},
		{"FP32", serializeFloat32([]float32{1.0, 2.0}), []float32{1.0, 2.0}, false},
		{"FP64", serializeFloat64([]float64{1.0, 2.0}), []float64{1.0, 2.0}, false},
		{"BYTES", serializeStrings([]string{"hello", "world"}), []string{"hello", "world"}, false},
		{"BF16", serializeBF16([]float32{1.0, 2.0}), []float32{1.0, 2.0}, false},
		{"UNSUPPORTED", nil, nil, true},
	}
	for _, tc := range testCases {
		result, err := dc.DeserializeTensor(tc.datatype, tc.dataBuffer)
		if (err != nil) != tc.expectErr {
			t.Errorf("DeserializeTensor(%v, %v) unexpected error status: %v", tc.datatype, tc.dataBuffer, err)
		}
		if !tc.expectErr && !reflect.DeepEqual(result, tc.expected) {
			t.Errorf("DeserializeTensor(%v, %v) = %v; want %v", tc.datatype, tc.dataBuffer, result, tc.expected)
		}
	}
}

func TestSerializeTensor_Error(t *testing.T) {
	dc := &dataConverter{}

	testCases := []struct {
		description string
		input       interface{}
		expected    []byte
		expectError bool
		errorMsg    string
	}{
		{
			description: "Serialize []int",
			input:       []int{1, 2, 3},
			expected: func() []byte {
				buf := new(bytes.Buffer)
				for _, v := range []int{1, 2, 3} {
					binary.Write(buf, binary.LittleEndian, int64(v))
				}
				return buf.Bytes()
			}(),
			expectError: false,
		},
		{
			description: "Serialize []int32",
			input:       []int32{1, 2, 3},
			expected: func() []byte {
				buf := new(bytes.Buffer)
				for _, v := range []int32{1, 2, 3} {
					binary.Write(buf, binary.LittleEndian, v)
				}
				return buf.Bytes()
			}(),
			expectError: false,
		},
		{
			description: "Unsupported tensor datatype",
			input:       []complex64{1 + 2i},
			expectError: true,
			errorMsg:    "unsupported tensor datatype",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			result, err := dc.SerializeTensor(tc.input)
			if tc.expectError {
				if err == nil || err.Error() != tc.errorMsg {
					t.Errorf("Expected error '%v', got '%v'", tc.errorMsg, err)
				}
				return
			}
			if err != nil {
				t.Errorf("Unexpected error: %v", err)
			}
			if !bytes.Equal(result, tc.expected) {
				t.Errorf("Expected %v, got %v", tc.expected, result)
			}
		})
	}

	t.Run("Simulate write error int", func(t *testing.T) {
		faultyWriter := &FaultyWriter{failOnWrite: true}
		err := dc.serializeTensorToWriter([]int{1, 2, 3}, faultyWriter)
		if err == nil || err.Error() != "write error" {
			t.Errorf("Expected 'write error', got '%v'", err)
		}
	})

	t.Run("Simulate write error int32", func(t *testing.T) {
		faultyWriter := &FaultyWriter{failOnWrite: true}
		err := dc.serializeTensorToWriter([]int32{1, 2, 3}, faultyWriter)
		if err == nil || err.Error() != "write error" {
			t.Errorf("Expected 'write error', got '%v'", err)
		}
	})

	t.Run("Simulate write error int64", func(t *testing.T) {
		faultyWriter := &FaultyWriter{failOnWrite: true}
		err := dc.serializeTensorToWriter([]int64{1, 2, 3}, faultyWriter)
		if err == nil || err.Error() != "write error" {
			t.Errorf("Expected 'write error', got '%v'", err)
		}
	})

	t.Run("Simulate write error uint16", func(t *testing.T) {
		faultyWriter := &FaultyWriter{failOnWrite: true}
		err := dc.serializeTensorToWriter([]uint16{1, 2, 3}, faultyWriter)
		if err == nil || err.Error() != "write error" {
			t.Errorf("Expected 'write error', got '%v'", err)
		}
	})

	t.Run("Simulate write error uint32", func(t *testing.T) {
		faultyWriter := &FaultyWriter{failOnWrite: true}
		err := dc.serializeTensorToWriter([]uint32{1, 2, 3}, faultyWriter)
		if err == nil || err.Error() != "write error" {
			t.Errorf("Expected 'write error', got '%v'", err)
		}
	})

	t.Run("Simulate write error uint64", func(t *testing.T) {
		faultyWriter := &FaultyWriter{failOnWrite: true}
		err := dc.serializeTensorToWriter([]uint64{1, 2, 3}, faultyWriter)
		if err == nil || err.Error() != "write error" {
			t.Errorf("Expected 'write error', got '%v'", err)
		}
	})

	t.Run("Simulate write error float32", func(t *testing.T) {
		faultyWriter := &FaultyWriter{failOnWrite: true}
		err := dc.serializeTensorToWriter([]float32{1, 2, 3}, faultyWriter)
		if err == nil || err.Error() != "write error" {
			t.Errorf("Expected 'write error', got '%v'", err)
		}
	})

	t.Run("Simulate write error float64", func(t *testing.T) {
		faultyWriter := &FaultyWriter{failOnWrite: true}
		err := dc.serializeTensorToWriter([]float64{1, 2, 3}, faultyWriter)
		if err == nil || err.Error() != "write error" {
			t.Errorf("Expected 'write error', got '%v'", err)
		}
	})

	t.Run("Simulate write error bool", func(t *testing.T) {
		faultyWriter := &FaultyWriter{failOnWrite: true}
		err := dc.serializeTensorToWriter([]bool{true, false}, faultyWriter)
		if err == nil || err.Error() != "write error" {
			t.Errorf("Expected 'write error', got '%v'", err)
		}
	})

	t.Run("Simulate write error byte", func(t *testing.T) {
		faultyWriter := &FaultyWriter{failOnWrite: true}
		err := dc.serializeTensorToWriter([]byte{1, 4, 5}, faultyWriter)
		if err == nil || err.Error() != "write error" {
			t.Errorf("Expected 'write error', got '%v'", err)
		}
	})

	t.Run("Simulate write error string", func(t *testing.T) {
		faultyWriter := &FaultyWriter{failOnWrite: true}
		err := dc.serializeTensorToWriter([]string{"test1", "test2"}, faultyWriter)
		if err == nil || err.Error() != "write error" {
			t.Errorf("Expected 'write error', got '%v'", err)
		}
	})

	t.Run("Simulate write error with []string", func(t *testing.T) {
		str := "errorString"
		tensor := []string{str, "test"}

		buffer := &bytes.Buffer{}
		failingWriter := &FaultyStringWriter{
			underlyingWriter: buffer,
			failOnData:       []byte(str),
		}

		err := dc.serializeTensorToWriter(tensor, failingWriter)
		if err == nil || err.Error() != "write error" {
			t.Errorf("Expected 'write error', got '%v'", err)
		}
	})
}

func TestFlattenData(t *testing.T) {
	dc := NewDataConverter()
	testCases := []struct {
		input    interface{}
		expected []interface{}
	}{
		{[]int{1, 2, 3}, []interface{}{1, 2, 3}},
		{[]int32{1, 2, 3}, []interface{}{int32(1), int32(2), int32(3)}},
		{[]int64{1, 2, 3}, []interface{}{int64(1), int64(2), int64(3)}},
		{[]uint16{1, 2, 3}, []interface{}{uint16(1), uint16(2), uint16(3)}},
		{[]uint32{1, 2, 3}, []interface{}{uint32(1), uint32(2), uint32(3)}},
		{[]uint64{1, 2, 3}, []interface{}{uint64(1), uint64(2), uint64(3)}},
		{[]float32{1.0, 2.0, 3.0}, []interface{}{float32(1.0), float32(2.0), float32(3.0)}},
		{[]float64{1.0, 2.0, 3.0}, []interface{}{float64(1.0), float64(2.0), float64(3.0)}},
		{[]byte{0x01, 0x02}, []interface{}{byte(0x01), byte(0x02)}},
		{[]bool{true, false}, []interface{}{true, false}},
		{[]string{"hello", "world"}, []interface{}{"hello", "world"}},
		{[]struct{}{}, nil},
	}
	for _, tc := range testCases {
		result := dc.FlattenData(tc.input)
		if !reflect.DeepEqual(result, tc.expected) {
			t.Errorf("FlattenData(%v) = %v; want %v", tc.input, result, tc.expected)
		}
	}
}

func TestReshapeArray(t *testing.T) {
	dc := NewDataConverter()
	data := []interface{}{1, 2, 3, 4, 5, 6}
	shape := []int{2, 3}
	expected := []interface{}{
		[]interface{}{1, 2, 3},
		[]interface{}{4, 5, 6},
	}
	result, err := dc.ReshapeArray(data, shape)
	if err != nil {
		t.Errorf("ReshapeArray(%v, %v) unexpected error: %v", data, shape, err)
	}
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("ReshapeArray(%v, %v) = %v; want %v", data, shape, result, expected)
	}
	_, err = dc.ReshapeArray(data, []int{})
	if err == nil {
		t.Errorf("Expected error for empty shape")
	}
	_, err = dc.ReshapeArray(data, []int{2, 2})
	if err == nil {
		t.Errorf("Expected error for size mismatch")
	}
}

func TestConvertByteSliceToInt32Slice(t *testing.T) {
	dc := NewDataConverter()
	data := []byte{1, 2, 3, 4}
	expected := []int32{1, 2, 3, 4}
	result := dc.ConvertByteSliceToInt32Slice(data)
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("ConvertByteSliceToInt32Slice(%v) = %v; want %v", data, result, expected)
	}
}

func TestConvertByteSliceToInt64Slice(t *testing.T) {
	dc := NewDataConverter()
	data := make([]byte, 16)
	binary.LittleEndian.PutUint64(data[0:], uint64(1))
	binary.LittleEndian.PutUint64(data[8:], uint64(2))
	expected := []int64{1, 2}
	result := dc.ConvertByteSliceToInt64Slice(data)
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("ConvertByteSliceToInt64Slice(%v) = %v; want %v", data, result, expected)
	}
}

func TestConvertByteSliceToFloat32Slice(t *testing.T) {
	dc := NewDataConverter()
	data := make([]byte, 8)
	binary.LittleEndian.PutUint32(data[0:], math.Float32bits(1.0))
	binary.LittleEndian.PutUint32(data[4:], math.Float32bits(2.0))
	expected := []float32{1.0, 2.0}
	result := dc.ConvertByteSliceToFloat32Slice(data)
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("ConvertByteSliceToFloat32Slice(%v) = %v; want %v", data, result, expected)
	}
}

func TestConvertByteSliceToFloat64Slice(t *testing.T) {
	dc := NewDataConverter()
	data := make([]byte, 16)
	binary.LittleEndian.PutUint64(data[0:], math.Float64bits(1.0))
	binary.LittleEndian.PutUint64(data[8:], math.Float64bits(2.0))
	expected := []float64{1.0, 2.0}
	result := dc.ConvertByteSliceToFloat64Slice(data)
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("ConvertByteSliceToFloat64Slice(%v) = %v; want %v", data, result, expected)
	}
}

func TestConvertInterfaceSliceToFloat32SliceAsInterface(t *testing.T) {
	dc := NewDataConverter()
	data := []interface{}{float64(1.0), float64(2.0)}
	expected := []interface{}{float32(1.0), float32(2.0)}
	result := dc.ConvertInterfaceSliceToFloat32SliceAsInterface(data)
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("ConvertInterfaceSliceToFloat32SliceAsInterface(%v) = %v; want %v", data, result, expected)
	}
}

func TestConvertInterfaceSliceToFloat64SliceAsInterface(t *testing.T) {
	dc := NewDataConverter()
	data := []interface{}{float64(1.0), float64(2.0)}
	expected := []interface{}{float64(1.0), float64(2.0)}
	result := dc.ConvertInterfaceSliceToFloat64SliceAsInterface(data)
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("ConvertInterfaceSliceToFloat64SliceAsInterface(%v) = %v; want %v", data, result, expected)
	}
}

func TestConvertInterfaceSliceToInt32SliceAsInterface(t *testing.T) {
	dc := NewDataConverter()
	data := []interface{}{float64(1), float64(2)}
	expected := []interface{}{int32(1), int32(2)}
	result := dc.ConvertInterfaceSliceToInt32SliceAsInterface(data)
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("ConvertInterfaceSliceToInt32SliceAsInterface(%v) = %v; want %v", data, result, expected)
	}
}

func TestConvertInterfaceSliceToInt64SliceAsInterface(t *testing.T) {
	dc := NewDataConverter()
	data := []interface{}{float64(1), float64(2)}
	expected := []interface{}{int64(1), int64(2)}
	result := dc.ConvertInterfaceSliceToInt64SliceAsInterface(data)
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("ConvertInterfaceSliceToInt64SliceAsInterface(%v) = %v; want %v", data, result, expected)
	}
}

func TestConvertInterfaceSliceToUint32SliceAsInterface(t *testing.T) {
	dc := NewDataConverter()
	data := []interface{}{float64(1), float64(2)}
	expected := []interface{}{uint32(1), uint32(2)}
	result := dc.ConvertInterfaceSliceToUint32SliceAsInterface(data)
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("ConvertInterfaceSliceToUint32SliceAsInterface(%v) = %v; want %v", data, result, expected)
	}
}

func TestConvertInterfaceSliceToUint64SliceAsInterface(t *testing.T) {
	dc := NewDataConverter()
	data := []interface{}{float64(1), float64(2)}
	expected := []interface{}{uint64(1), uint64(2)}
	result := dc.ConvertInterfaceSliceToUint64SliceAsInterface(data)
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("ConvertInterfaceSliceToUint64SliceAsInterface(%v) = %v; want %v", data, result, expected)
	}
}

func TestConvertInterfaceSliceToBoolSliceAsInterface(t *testing.T) {
	dc := NewDataConverter()
	data := []interface{}{true, false}
	expected := []interface{}{true, false}
	result := dc.ConvertInterfaceSliceToBoolSliceAsInterface(data)
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("ConvertInterfaceSliceToBoolSliceAsInterface(%v) = %v; want %v", data, result, expected)
	}
}

func TestConvertInterfaceSliceToBytesSliceAsInterface(t *testing.T) {
	dc := NewDataConverter()
	data := []interface{}{"hello", "world"}
	expected := []interface{}{[]byte("hello"), []byte("world")}
	result, err := dc.ConvertInterfaceSliceToBytesSliceAsInterface(data)
	if err != nil {
		t.Errorf("ConvertInterfaceSliceToBytesSliceAsInterface(%v) unexpected error: %v", data, err)
	}
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("ConvertInterfaceSliceToBytesSliceAsInterface(%v) = %v; want %v", data, result, expected)
	}
	data = []interface{}{"hello", 123}
	_, err = dc.ConvertInterfaceSliceToBytesSliceAsInterface(data)
	if err == nil {
		t.Errorf("Expected error for non-string data")
	}
}

func TestDeserializeBytesTensor(t *testing.T) {
	dc := dataConverter{}
	data := serializeStrings([]string{"hello", "world"})
	result, err := dc.deserializeBytesTensor(data)
	expected := []string{"hello", "world"}
	if err != nil {
		t.Errorf("deserializeBytesTensor(%v) unexpected error: %v", data, err)
	}
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("deserializeBytesTensor(%v) = %v; want %v", data, result, expected)
	}
	data = []byte{0x05, 0x00, 0x00}
	_, err = dc.deserializeBytesTensor(data)
	if err == nil {
		t.Errorf("Expected error for malformed data")
	}
}

func TestReshapeArray_ErrorHandling(t *testing.T) {
	dc := &dataConverter{}

	testCases := []struct {
		description string
		data        interface{}
		shape       []int
		expectError bool
		errorMsg    string
	}{
		{
			description: "Invalid shape with zero dimension",
			data:        []int{1, 2, 3, 4},
			shape:       []int{2, 0},
			expectError: true,
			errorMsg:    "invalid shape: dimensions must be positive integers",
		},
		{
			description: "Shape not matching data length",
			data:        []int{1, 2, 3, 4},
			shape:       []int{2, 2, 2},
			expectError: true,
			errorMsg:    "data length does not match shape",
		},
		{
			description: "Data cannot be evenly divided according to the shape",
			data:        []int{1, 2, 3, 4, 5},
			shape:       []int{2, 2},
			expectError: true,
			errorMsg:    "data cannot be evenly divided according to the shape",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			_, err := dc.ReshapeArray(tc.data, tc.shape)
			if tc.expectError {
				if err == nil || !strings.Contains(err.Error(), tc.errorMsg) {
					t.Errorf("Expected error containing '%v', got '%v'", tc.errorMsg, err)
				}
			} else if err != nil {
				t.Errorf("Unexpected error: %v", err)
			}
		})
	}
}

func TestReshapeRecursively_EmptyShape(t *testing.T) {
	dc := &dataConverter{}
	data := []int{1, 2, 3, 4}
	rv := reflect.ValueOf(data)
	shape := []int{}

	_, err := dc.reshapeRecursively(rv, shape)
	if err == nil || err.Error() != "shape cannot be empty" {
		t.Errorf("Expected error 'shape cannot be empty', got '%v'", err)
	}
}

func TestDeserializeBytesTensor_UnexpectedEnd(t *testing.T) {
	dc := &dataConverter{}

	length := uint32(5)
	buf := make([]byte, 4)
	binary.LittleEndian.PutUint32(buf, length)
	buf = append(buf, []byte("abc")...)

	_, err := dc.deserializeBytesTensor(buf)
	if err == nil || err.Error() != "unexpected end of encoded tensor" {
		t.Errorf("Expected 'unexpected end of encoded tensor', got '%v'", err)
	}
}
