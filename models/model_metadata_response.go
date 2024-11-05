package models

type ModelMetadataResponse struct {
	Name     string                `json:"name"`
	Versions []string              `json:"versions"`
	Platform string                `json:"platform"`
	Inputs   []ModelMetadataInput  `json:"inputs"`
	Outputs  []ModelMetadataOutput `json:"outputs"`
}

type ModelMetadataInput struct {
	Name     string `json:"name"`
	Datatype string `json:"datatype"`
	Shape    []int  `json:"shape"`
}

type ModelMetadataOutput struct {
	Name     string `json:"name"`
	Datatype string `json:"datatype"`
	Shape    []int  `json:"shape"`
}
