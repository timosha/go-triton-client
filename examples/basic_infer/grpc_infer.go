package basic_infer

import (
	"context"
	"fmt"
	"github.com/Trendyol/go-triton-client/base"
	"github.com/Trendyol/go-triton-client/client/grpc"
	"github.com/Trendyol/go-triton-client/parser"
	"github.com/Trendyol/go-triton-client/postprocess"
	"log"
)

func performGRPCInference(tritonClient base.Client) {
	input := grpc.NewInferInput("input_ids", "INT64", []int64{1, 1}, nil)
	err := input.SetData([]int{101, 202536, 102}, true) // Example data
	if err != nil {
		log.Fatal(err)
	}

	outputs := []base.InferOutput{
		grpc.NewInferOutput("output_name", map[string]interface{}{"binary_data": true}),
	}

	response, err := tritonClient.Infer(
		context.Background(),
		"model_name",
		"model_version",
		[]base.InferInput{input},
		outputs,
		nil,
	)
	if err != nil {
		log.Fatal(err)
	}

	// Process response
	sliceResp, err := response.AsSlice("output_name")
	if err != nil {
		log.Fatal(err)
	}

	// Parse the response
	parsedData, ok := parser.ParseSlice[[][]float64](sliceResp)
	if !ok {
		log.Fatal("Failed to parse inference response")
	}

	// Post-process if needed
	postprocessManager := postprocess.NewPostprocessManager()
	processedData := postprocessManager.Float64ToFloat32Slice2D(parsedData)
	fmt.Println(processedData)
}
