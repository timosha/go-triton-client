package basic_infer

import (
	"context"
	"fmt"
	"github.com/Trendyol/go-triton-client/base"
	"github.com/Trendyol/go-triton-client/client/grpc"
	"github.com/Trendyol/go-triton-client/converter"
	"log"
)

func performGRPCInference(tritonClient base.Client) {
	input := grpc.NewInferInput("input_ids", "INT64", []int64{1, 1}, nil)
	err := input.SetData([]int{101, 202536, 102}, true) // Example data
	if err != nil {
		log.Fatal(err)
	}

	outputs := []base.InferOutput{
		grpc.NewInferOutput("output_name", map[string]any{"binary_data": true}),
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
	sliceResp, err := response.AsFloat32Slice("output_name")
	if err != nil {
		log.Fatal(err)
	}

	output, _ := response.GetOutput("logits")

	embeddings, err := converter.Reshape3D[float32](sliceResp, output.GetShape())
	if err != nil {
		log.Fatal("Failed to reshape inference response")
	}

	fmt.Println(embeddings)
}
