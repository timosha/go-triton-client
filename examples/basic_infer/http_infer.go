package basic_infer

import (
	"context"
	"fmt"
	"github.com/Trendyol/go-triton-client/base"
	"github.com/Trendyol/go-triton-client/client/http"
	"github.com/Trendyol/go-triton-client/parser"
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
		http.NewInferOutput("logits", map[string]interface{}{"binary_data": true}),
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

	sliceResp, err := response.AsSlice("logits")
	if err != nil {
		log.Fatal(err)
	}

	embeddings, ok := parser.ParseSlice[[][][]float64](sliceResp)
	if !ok {
		log.Fatal("Failed to parse inference response")
	}

	postprocessManager := postprocess.NewPostprocessManager()
	convertedEmbeddings := postprocessManager.Float64ToFloat32Slice3D(embeddings)

	meanPooledEmbeddings, err := postprocessManager.MeanPoolingFloat32Slice3D(convertedEmbeddings, [][]int64{{1, 1, 1}, {1, 1, 1}})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(meanPooledEmbeddings)
}
