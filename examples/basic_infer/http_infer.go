package basic_infer

import (
	"context"
	"fmt"
	"github.com/Trendyol/go-triton-client/base"
	"github.com/Trendyol/go-triton-client/client/http"
	"github.com/Trendyol/go-triton-client/converter"
	"github.com/Trendyol/go-triton-client/postprocess"
	"log"
)

func performHttpInference(tritonClient base.Client) {
	inputIds := http.NewInferInput("input_ids", "INT64", []int64{2, 3}, nil)
	err := inputIds.SetData([]int{101, 202536, 102, 101, 202536, 102}, true)
	if err != nil {
		log.Fatal(err)
	}
	tokenTypeIds := http.NewInferInput("token_type_ids", "INT64", []int64{2, 3}, nil)
	err = tokenTypeIds.SetData([]int{0, 0, 0, 0, 0, 0}, true)
	if err != nil {
		log.Fatal(err)
	}
	attentionMask := http.NewInferInput("attention_mask", "INT64", []int64{2, 3}, nil)
	err = attentionMask.SetData([]int{1, 1, 1, 1, 1, 1}, true)
	if err != nil {
		log.Fatal(err)
	}

	outputs := []base.InferOutput{
		http.NewInferOutput("logits", map[string]any{"binary_data": true}),
	}

	response, err := tritonClient.Infer(
		context.Background(),
		"ty_bert",
		"1",
		[]base.InferInput{inputIds, tokenTypeIds, attentionMask},
		outputs,
		nil,
	)
	if err != nil {
		log.Fatal(err)
	}
	sliceResp, err := response.AsFloat32Slice("logits")
	if err != nil {
		log.Fatal(err)
	}
	output, _ := response.GetOutput("logits")

	embeddings, err := converter.Reshape3D[float32](sliceResp, output.GetShape())
	if err != nil {
		log.Fatal("Failed to reshape inference response")
	}

	postprocessManager := postprocess.NewPostprocessManager()
	meanPooledEmbeddings, err := postprocessManager.MeanPoolingFloat32Slice3D(embeddings, [][]int64{{1, 1, 1}, {1, 1, 1}})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(meanPooledEmbeddings)
}
