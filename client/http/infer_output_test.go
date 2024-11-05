package http

import (
	"encoding/json"
	"errors"
	"github.com/Trendyol/go-triton-client/base"
	"github.com/Trendyol/go-triton-client/mocks"
	"reflect"
	"testing"

	"github.com/golang/mock/gomock"
)

func TestNewInferOutput(t *testing.T) {
	parameters := map[string]interface{}{"param1": "value1"}
	output := NewInferOutput("output_name", parameters)
	if output.GetName() != "output_name" {
		t.Errorf("Expected name 'output_name', got '%s'", output.GetName())
	}
	if output.GetParameters()["param1"] != "value1" {
		t.Errorf("Expected parameter 'param1' to be 'value1', got '%v'", output.GetParameters()["param1"])
	}
}

func TestNewInferOutput_NilParameters(t *testing.T) {
	output := NewInferOutput("output_name", nil)
	if output.GetParameters() == nil {
		t.Error("Expected parameters to be initialized, got nil")
	}
	if len(output.GetParameters()) != 0 {
		t.Errorf("Expected empty parameters, got %v", output.GetParameters())
	}
}

func TestInferOutput_GetTensor(t *testing.T) {
	parameters := map[string]interface{}{"param1": "value1"}
	output := NewInferOutput("output_name", parameters)
	tensor := output.GetTensor().(map[string]interface{})
	if tensor["name"] != "output_name" {
		t.Errorf("Expected tensor name 'output_name', got '%s'", tensor["name"])
	}
	if tensor["parameters"].(map[string]interface{})["param1"] != "value1" {
		t.Errorf("Expected parameter 'param1' to be 'value1', got '%v'", tensor["parameters"].(map[string]interface{})["param1"])
	}
}

func TestInferOutput_UnmarshalJSON_AllDataTypes(t *testing.T) {
	dataTypes := []string{"FP32", "FP64", "INT32", "INT64", "UINT32", "UINT64", "BOOL", "BYTES", "UNKNOWN"}
	for _, dtype := range dataTypes {
		t.Run("Datatype_"+dtype, func(t *testing.T) {
			mockController := gomock.NewController(t)
			defer mockController.Finish()
			mockDataConverter := mocks.NewMockDataConverter(mockController)

			jsonData := `{
				"name": "output_name",
				"shape": [2],
				"datatype": "` + dtype + `",
				"parameters": {"param1": "value1"},
				"data": [1, 2]
			}`
			output := &InferOutput{
				BaseInferOutput: &base.BaseInferOutput{},
				DataConverter:   mockDataConverter,
			}

			switch dtype {
			case "FP32":
				mockDataConverter.EXPECT().ConvertInterfaceSliceToFloat32SliceAsInterface(gomock.Any()).Return([]interface{}{1.0, 2.0})
			case "FP64":
				mockDataConverter.EXPECT().ConvertInterfaceSliceToFloat64SliceAsInterface(gomock.Any()).Return([]interface{}{1.0, 2.0})
			case "INT32":
				mockDataConverter.EXPECT().ConvertInterfaceSliceToInt32SliceAsInterface(gomock.Any()).Return([]interface{}{1, 2})
			case "INT64":
				mockDataConverter.EXPECT().ConvertInterfaceSliceToInt64SliceAsInterface(gomock.Any()).Return([]interface{}{1, 2})
			case "UINT32":
				mockDataConverter.EXPECT().ConvertInterfaceSliceToUint32SliceAsInterface(gomock.Any()).Return([]interface{}{1, 2})
			case "UINT64":
				mockDataConverter.EXPECT().ConvertInterfaceSliceToUint64SliceAsInterface(gomock.Any()).Return([]interface{}{1, 2})
			case "BOOL":
				mockDataConverter.EXPECT().ConvertInterfaceSliceToBoolSliceAsInterface(gomock.Any()).Return([]interface{}{true, false})
			case "BYTES":
				mockDataConverter.EXPECT().ConvertInterfaceSliceToBytesSliceAsInterface(gomock.Any()).Return([]interface{}{[]byte("data1"), []byte("data2")}, nil)
			default:
				// No conversion for UNKNOWN type
			}

			err := json.Unmarshal([]byte(jsonData), output)
			if err != nil {
				t.Errorf("Unexpected error: %v", err)
			}

			switch dtype {
			case "FP32":
				expectedData := []interface{}{1.0, 2.0}
				if !reflect.DeepEqual(output.Data, expectedData) {
					t.Errorf("Expected data %v, got %v", expectedData, output.Data)
				}
			case "FP64":
				expectedData := []interface{}{1.0, 2.0}
				if !reflect.DeepEqual(output.Data, expectedData) {
					t.Errorf("Expected data %v, got %v", expectedData, output.Data)
				}
			case "INT32":
				expectedData := []interface{}{1, 2}
				if !reflect.DeepEqual(output.Data, expectedData) {
					t.Errorf("Expected data %v, got %v", expectedData, output.Data)
				}
			case "INT64":
				expectedData := []interface{}{1, 2}
				if !reflect.DeepEqual(output.Data, expectedData) {
					t.Errorf("Expected data %v, got %v", expectedData, output.Data)
				}
			case "UINT32":
				expectedData := []interface{}{1, 2}
				if !reflect.DeepEqual(output.Data, expectedData) {
					t.Errorf("Expected data %v, got %v", expectedData, output.Data)
				}
			case "UINT64":
				expectedData := []interface{}{1, 2}
				if !reflect.DeepEqual(output.Data, expectedData) {
					t.Errorf("Expected data %v, got %v", expectedData, output.Data)
				}
			case "BOOL":
				expectedData := []interface{}{true, false}
				if !reflect.DeepEqual(output.Data, expectedData) {
					t.Errorf("Expected data %v, got %v", expectedData, output.Data)
				}
			case "BYTES":
				expectedData := []interface{}{[]byte("data1"), []byte("data2")}
				if !reflect.DeepEqual(output.Data, expectedData) {
					t.Errorf("Expected data %v, got %v", expectedData, output.Data)
				}
			default:
				expectedData := []interface{}{1.0, 2.0}
				if !reflect.DeepEqual(output.Data, expectedData) {
					t.Errorf("Expected data %v, got %v", expectedData, output.Data)
				}
			}
		})
	}
}

func TestInferOutput_UnmarshalJSON_BytesConversionError(t *testing.T) {
	mockController := gomock.NewController(t)
	defer mockController.Finish()
	mockDataConverter := mocks.NewMockDataConverter(mockController)

	jsonData := `{
		"name": "output_name",
		"shape": [2],
		"datatype": "BYTES",
		"parameters": {"param1": "value1"},
		"data": ["invalid data"]
	}`
	output := &InferOutput{
		BaseInferOutput: &base.BaseInferOutput{},
		DataConverter:   mockDataConverter,
	}

	mockDataConverter.EXPECT().ConvertInterfaceSliceToBytesSliceAsInterface(gomock.Any()).Return(nil, errors.New("conversion error"))

	err := json.Unmarshal([]byte(jsonData), output)
	if err == nil {
		t.Errorf("Expected error, got nil")
	}
	if err.Error() != "conversion error" {
		t.Errorf("Expected error 'conversion error', got '%v'", err)
	}
}

func TestInferOutput_UnmarshalJSON_UnmarshalError(t *testing.T) {
	mockController := gomock.NewController(t)
	defer mockController.Finish()
	mockDataConverter := mocks.NewMockDataConverter(mockController)

	jsonData := `{
        "name": "output_name",
        "datatype": "FP32",
        "data": [1.0, 2.0],
        "shape": [2],
        "classification": "invalid_int"
    }`

	output := &InferOutput{
		BaseInferOutput: &base.BaseInferOutput{},
		DataConverter:   mockDataConverter,
	}

	err := json.Unmarshal([]byte(jsonData), output)
	if err == nil {
		t.Errorf("Expected error due to invalid type for 'classification', got nil")
	}
}

func TestInferOutput_UnmarshalJSON_NoParameters(t *testing.T) {
	mockController := gomock.NewController(t)
	defer mockController.Finish()
	mockDataConverter := mocks.NewMockDataConverter(mockController)

	jsonData := `{
		"name": "output_name",
		"shape": [2],
		"datatype": "INT32",
		"data": [1, 2]
	}`
	output := &InferOutput{
		BaseInferOutput: &base.BaseInferOutput{},
		DataConverter:   mockDataConverter,
	}

	mockDataConverter.EXPECT().ConvertInterfaceSliceToInt32SliceAsInterface(gomock.Any()).Return([]interface{}{1, 2})

	err := json.Unmarshal([]byte(jsonData), output)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if output.Parameters != nil {
		t.Errorf("Expected nil parameters, got %v", output.Parameters)
	}
	expectedData := []interface{}{1, 2}
	if !reflect.DeepEqual(output.Data, expectedData) {
		t.Errorf("Expected data %v, got %v", expectedData, output.Data)
	}
}

func TestInferOutput_UnmarshalJSON_EmptyData(t *testing.T) {
	mockController := gomock.NewController(t)
	defer mockController.Finish()
	mockDataConverter := mocks.NewMockDataConverter(mockController)

	jsonData := `{
		"name": "output_name",
		"shape": [2],
		"datatype": "INT32",
		"parameters": {"param1": "value1"},
		"data": []
	}`
	output := &InferOutput{
		BaseInferOutput: &base.BaseInferOutput{},
		DataConverter:   mockDataConverter,
	}

	mockDataConverter.EXPECT().ConvertInterfaceSliceToInt32SliceAsInterface(gomock.Any()).Return([]interface{}{})

	err := json.Unmarshal([]byte(jsonData), output)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	expectedData := []interface{}{}
	if !reflect.DeepEqual(output.Data, expectedData) {
		t.Errorf("Expected empty data, got %v", output.Data)
	}
}

func TestInferOutput_UnmarshalJSON_MissingDataField(t *testing.T) {
	mockController := gomock.NewController(t)
	defer mockController.Finish()
	mockDataConverter := mocks.NewMockDataConverter(mockController)
	mockDataConverter.EXPECT().ConvertInterfaceSliceToInt32SliceAsInterface(gomock.Any()).Return(nil)

	jsonData := `{
		"name": "output_name",
		"shape": [2],
		"datatype": "INT32",
		"parameters": {"param1": "value1"}
	}`
	output := &InferOutput{
		BaseInferOutput: &base.BaseInferOutput{},
		DataConverter:   mockDataConverter,
	}

	err := json.Unmarshal([]byte(jsonData), output)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if output.Data != nil {
		t.Errorf("Expected nil data, got %v", output.Data)
	}
}

func TestInferOutput_UnmarshalJSON_MissingShapeField(t *testing.T) {
	mockController := gomock.NewController(t)
	defer mockController.Finish()
	mockDataConverter := mocks.NewMockDataConverter(mockController)

	jsonData := `{
		"name": "output_name",
		"datatype": "INT32",
		"parameters": {"param1": "value1"},
		"data": [1, 2]
	}`
	output := &InferOutput{
		BaseInferOutput: &base.BaseInferOutput{},
		DataConverter:   mockDataConverter,
	}

	mockDataConverter.EXPECT().ConvertInterfaceSliceToInt32SliceAsInterface(gomock.Any()).Return([]interface{}{1, 2})

	err := json.Unmarshal([]byte(jsonData), output)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if output.Shape != nil {
		t.Errorf("Expected nil shape, got %v", output.Shape)
	}
}

func TestInferOutput_UnmarshalJSON_MissingNameField(t *testing.T) {
	mockController := gomock.NewController(t)
	defer mockController.Finish()
	mockDataConverter := mocks.NewMockDataConverter(mockController)

	jsonData := `{
		"shape": [2],
		"datatype": "INT32",
		"parameters": {"param1": "value1"},
		"data": [1, 2]
	}`
	output := &InferOutput{
		BaseInferOutput: &base.BaseInferOutput{},
		DataConverter:   mockDataConverter,
	}

	mockDataConverter.EXPECT().ConvertInterfaceSliceToInt32SliceAsInterface(gomock.Any()).Return([]interface{}{1, 2})

	err := json.Unmarshal([]byte(jsonData), output)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if output.Name != "" {
		t.Errorf("Expected empty name, got '%s'", output.Name)
	}
}

func TestInferOutput_UnmarshalJSON_MissingDatatypeField(t *testing.T) {
	mockController := gomock.NewController(t)
	defer mockController.Finish()
	mockDataConverter := mocks.NewMockDataConverter(mockController)

	jsonData := `{
		"name": "output_name",
		"shape": [2],
		"parameters": {"param1": "value1"},
		"data": [1, 2]
	}`
	output := &InferOutput{
		BaseInferOutput: &base.BaseInferOutput{},
		DataConverter:   mockDataConverter,
	}

	err := json.Unmarshal([]byte(jsonData), output)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if output.Datatype != "" {
		t.Errorf("Expected empty datatype, got '%s'", output.Datatype)
	}
	expectedData := []interface{}{1.0, 2.0}
	if !reflect.DeepEqual(output.Data, expectedData) {
		t.Errorf("Expected data %v, got %v", expectedData, output.Data)
	}
}
