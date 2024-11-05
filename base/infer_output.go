package base

import (
	"errors"
)

// InferOutputs holds model name, version, and output details.
type InferOutputs struct {
	ModelName    string             `json:"model_name"`
	ModelVersion string             `json:"model_version"`
	Outputs      []*BaseInferOutput `json:"outputs"`
}

// InferOutput interface defines methods for output data.
type InferOutput interface {
	GetName() string
	GetShape() []int
	GetDatatype() string
	GetParameters() map[string]interface{}
	GetData() []interface{}
	GetTensor() any
}

// BaseInferOutput represents basic output properties.
type BaseInferOutput struct {
	Name       string
	Shape      []int
	Datatype   string
	Parameters map[string]interface{}
	Data       []interface{}
}

// GetName returns the output name.
func (output *BaseInferOutput) GetName() string {
	return output.Name
}

// GetShape returns the output shape.
func (output *BaseInferOutput) GetShape() []int {
	return output.Shape
}

// GetDatatype returns the datatype of the output.
func (output *BaseInferOutput) GetDatatype() string {
	return output.Datatype
}

// GetParameters returns the parameters of the output.
func (output *BaseInferOutput) GetParameters() map[string]interface{} {
	return output.Parameters
}

// GetData returns the data of the output.
func (output *BaseInferOutput) GetData() []interface{} {
	return output.Data
}

// GetTensor returns an error indicating base function should not be used.
func (output *BaseInferOutput) GetTensor() any {
	return errors.New("do not use base GetTensor function")
}
