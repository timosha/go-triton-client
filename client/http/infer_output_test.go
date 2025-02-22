package http

import (
	"encoding/json"
	"github.com/Trendyol/go-triton-client/base"
	"reflect"
	"testing"

	"go.uber.org/mock/gomock"
)

func TestNewInferOutput(t *testing.T) {
	parameters := map[string]any{"param1": "value1"}
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
	parameters := map[string]any{"param1": "value1"}
	output := NewInferOutput("output_name", parameters)
	tensor := output.GetTensor().(map[string]any)
	if tensor["name"] != "output_name" {
		t.Errorf("Expected tensor name 'output_name', got '%s'", tensor["name"])
	}
	if tensor["parameters"].(map[string]any)["param1"] != "value1" {
		t.Errorf("Expected parameter 'param1' to be 'value1', got '%v'", tensor["parameters"].(map[string]any)["param1"])
	}
}

func TestInferOutput_UnmarshalJSON_AllDataTypes(t *testing.T) {
	dataTypes := []string{"FP32", "FP64", "INT32", "INT64", "UINT32", "UINT64", "BOOL", "BYTES", "UNKNOWN"}
	for _, dtype := range dataTypes {
		t.Run("Datatype_"+dtype, func(t *testing.T) {
			mockController := gomock.NewController(t)
			defer mockController.Finish()

			switch dtype {
			case "FP32":
				jsonData := `{
				"name": "output_name",
				"shape": [2],
				"datatype": "` + dtype + `",
				"parameters": {"param1": "value1"},
				"data": [1, 2]
			}`
				output := &InferOutput{
					BaseInferOutput: &base.BaseInferOutput{},
				}
				err := json.Unmarshal([]byte(jsonData), output)
				if err != nil {
					t.Errorf("Unexpected error: %v", err)
				}
				expectedData := []any{float32(1.0), float32(2.0)}
				if !reflect.DeepEqual(output.Data, expectedData) {
					t.Errorf("Expected data %v, got %v", expectedData, output.Data)
				}
			case "FP64":
				jsonData := `{
				"name": "output_name",
				"shape": [2],
				"datatype": "` + dtype + `",
				"parameters": {"param1": "value1"},
				"data": [1, 2]
			}`
				output := &InferOutput{
					BaseInferOutput: &base.BaseInferOutput{},
				}
				err := json.Unmarshal([]byte(jsonData), output)
				if err != nil {
					t.Errorf("Unexpected error: %v", err)
				}
				expectedData := []any{1.0, 2.0}
				if !reflect.DeepEqual(output.Data, expectedData) {
					t.Errorf("Expected data %v, got %v", expectedData, output.Data)
				}
			case "INT32":
				jsonData := `{
				"name": "output_name",
				"shape": [2],
				"datatype": "` + dtype + `",
				"parameters": {"param1": "value1"},
				"data": [1, 2]
			}`
				output := &InferOutput{
					BaseInferOutput: &base.BaseInferOutput{},
				}
				err := json.Unmarshal([]byte(jsonData), output)
				if err != nil {
					t.Errorf("Unexpected error: %v", err)
				}
				expectedData := []any{int32(1), int32(2)}
				if !reflect.DeepEqual(output.Data, expectedData) {
					t.Errorf("Expected data %v, got %v", expectedData, output.Data)
				}
			case "INT64":
				jsonData := `{
				"name": "output_name",
				"shape": [2],
				"datatype": "` + dtype + `",
				"parameters": {"param1": "value1"},
				"data": [1, 2]
			}`
				output := &InferOutput{
					BaseInferOutput: &base.BaseInferOutput{},
				}
				err := json.Unmarshal([]byte(jsonData), output)
				if err != nil {
					t.Errorf("Unexpected error: %v", err)
				}
				expectedData := []any{int64(1), int64(2)}
				if !reflect.DeepEqual(output.Data, expectedData) {
					t.Errorf("Expected data %v, got %v", expectedData, output.Data)
				}
			case "UINT32":
				jsonData := `{
				"name": "output_name",
				"shape": [2],
				"datatype": "` + dtype + `",
				"parameters": {"param1": "value1"},
				"data": [1, 2]
			}`
				output := &InferOutput{
					BaseInferOutput: &base.BaseInferOutput{},
				}
				err := json.Unmarshal([]byte(jsonData), output)
				if err != nil {
					t.Errorf("Unexpected error: %v", err)
				}
				expectedData := []any{uint32(1), uint32(2)}
				if !reflect.DeepEqual(output.Data, expectedData) {
					t.Errorf("Expected data %v, got %v", expectedData, output.Data)
				}
			case "UINT64":
				jsonData := `{
				"name": "output_name",
				"shape": [2],
				"datatype": "` + dtype + `",
				"parameters": {"param1": "value1"},
				"data": [1, 2]
			}`
				output := &InferOutput{
					BaseInferOutput: &base.BaseInferOutput{},
				}
				err := json.Unmarshal([]byte(jsonData), output)
				if err != nil {
					t.Errorf("Unexpected error: %v", err)
				}
				expectedData := []any{uint64(1), uint64(2)}
				if !reflect.DeepEqual(output.Data, expectedData) {
					t.Errorf("Expected data %v, got %v", expectedData, output.Data)
				}
			case "BOOL":
				jsonData := `{
				"name": "output_name",
				"shape": [2],
				"datatype": "` + dtype + `",
				"parameters": {"param1": "value1"},
				"data": [true, false]
			}`
				output := &InferOutput{
					BaseInferOutput: &base.BaseInferOutput{},
				}
				err := json.Unmarshal([]byte(jsonData), output)
				if err != nil {
					t.Errorf("Unexpected error: %v", err)
				}
				expectedData := []any{true, false}
				if !reflect.DeepEqual(output.Data, expectedData) {
					t.Errorf("Expected data %v, got %v", expectedData, output.Data)
				}
			case "BYTES":
				jsonData := `{
				"name": "output_name",
				"shape": [1],
				"datatype": "` + dtype + `",
				"parameters": {"param1": "value1"},
				"data": ["foo"]
			}`
				output := &InferOutput{
					BaseInferOutput: &base.BaseInferOutput{},
				}
				err := json.Unmarshal([]byte(jsonData), output)
				if err != nil {
					t.Errorf("Unexpected error: %v", err)
				}
				expectedData := []any{[]byte("foo")}
				if !reflect.DeepEqual(output.Data, expectedData) {
					t.Errorf("Expected data %v, got %v", expectedData, output.Data)
				}
			default:
				jsonData := `{
				"name": "output_name",
				"shape": [2],
				"datatype": "` + dtype + `",
				"parameters": {"param1": "value1"},
				"data": [1, 2]
			}`
				output := &InferOutput{
					BaseInferOutput: &base.BaseInferOutput{},
				}
				err := json.Unmarshal([]byte(jsonData), output)
				if err != nil {
					t.Errorf("Unexpected error: %v", err)
				}
				expectedData := []any{1.0, 2.0}
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

	jsonData := `{
		"name": "output_name",
		"shape": [2],
		"datatype": "BYTES",
		"parameters": {"param1": "value1"},
		"data": [1,2]
	}`
	output := &InferOutput{
		BaseInferOutput: &base.BaseInferOutput{},
	}

	err := json.Unmarshal([]byte(jsonData), output)
	if err == nil {
		t.Errorf("Expected error, got nil")
	}
}

func TestInferOutput_UnmarshalJSON_UnmarshalError(t *testing.T) {
	mockController := gomock.NewController(t)
	defer mockController.Finish()

	jsonData := `{
        "name": "output_name",
        "datatype": "FP32",
        "data": [1.0, 2.0],
        "shape": [2],
        "classification": "invalid_int"
    }`

	output := &InferOutput{
		BaseInferOutput: &base.BaseInferOutput{},
	}

	err := json.Unmarshal([]byte(jsonData), output)
	if err == nil {
		t.Errorf("Expected error due to invalid type for 'classification', got nil")
	}
}

func TestInferOutput_UnmarshalJSON_NoParameters(t *testing.T) {
	mockController := gomock.NewController(t)
	defer mockController.Finish()

	jsonData := `{
		"name": "output_name",
		"shape": [2],
		"datatype": "INT32",
		"data": [1, 2]
	}`
	output := &InferOutput{
		BaseInferOutput: &base.BaseInferOutput{},
	}

	err := json.Unmarshal([]byte(jsonData), output)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if output.Parameters != nil {
		t.Errorf("Expected nil parameters, got %v", output.Parameters)
	}
	expectedData := []any{int32(1), int32(2)}
	if !reflect.DeepEqual(output.Data, expectedData) {
		t.Errorf("Expected data %v, got %v", expectedData, output.Data)
	}
}

func TestInferOutput_UnmarshalJSON_EmptyData(t *testing.T) {
	mockController := gomock.NewController(t)
	defer mockController.Finish()

	jsonData := `{
		"name": "output_name",
		"shape": [2],
		"datatype": "INT32",
		"parameters": {"param1": "value1"},
		"data": []
	}`
	output := &InferOutput{
		BaseInferOutput: &base.BaseInferOutput{},
	}

	err := json.Unmarshal([]byte(jsonData), output)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	expectedData := []any{}
	if !reflect.DeepEqual(output.Data, expectedData) {
		t.Errorf("Expected empty data, got %v", output.Data)
	}
}

func TestInferOutput_UnmarshalJSON_MissingDataField(t *testing.T) {
	mockController := gomock.NewController(t)
	defer mockController.Finish()

	jsonData := `{
		"name": "output_name",
		"shape": [2],
		"datatype": "INT32",
		"parameters": {"param1": "value1"}
	}`
	output := &InferOutput{
		BaseInferOutput: &base.BaseInferOutput{},
	}

	err := json.Unmarshal([]byte(jsonData), output)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if len(output.Data) != 0 {
		t.Errorf("Expected nil data, got %v", output.Data)
	}
}

func TestInferOutput_UnmarshalJSON_MissingShapeField(t *testing.T) {
	mockController := gomock.NewController(t)
	defer mockController.Finish()

	jsonData := `{
		"name": "output_name",
		"datatype": "INT32",
		"parameters": {"param1": "value1"},
		"data": [1, 2]
	}`
	output := &InferOutput{
		BaseInferOutput: &base.BaseInferOutput{},
	}

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

	jsonData := `{
		"shape": [2],
		"datatype": "INT32",
		"parameters": {"param1": "value1"},
		"data": [1, 2]
	}`
	output := &InferOutput{
		BaseInferOutput: &base.BaseInferOutput{},
	}

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

	jsonData := `{
		"name": "output_name",
		"shape": [2],
		"parameters": {"param1": "value1"},
		"data": [1, 2]
	}`
	output := &InferOutput{
		BaseInferOutput: &base.BaseInferOutput{},
	}

	err := json.Unmarshal([]byte(jsonData), output)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if output.Datatype != "" {
		t.Errorf("Expected empty datatype, got '%s'", output.Datatype)
	}
	expectedData := []any{1.0, 2.0}
	if !reflect.DeepEqual(output.Data, expectedData) {
		t.Errorf("Expected data %v, got %v", expectedData, output.Data)
	}
}
