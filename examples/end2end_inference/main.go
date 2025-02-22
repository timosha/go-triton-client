package main

import (
	"context"
	"fmt"
	"github.com/Trendyol/go-triton-client/converter"
	"log"

	"github.com/Trendyol/go-triton-client/base"
	triton "github.com/Trendyol/go-triton-client/client/http"
	"github.com/Trendyol/go-triton-client/postprocess"
	"github.com/Trendyol/go-triton-client/tokenizer"
	tokenizerOptions "github.com/Trendyol/go-triton-client/tokenizer/options"
)

func main() {
	// Initialize the Triton HTTP client
	tritonClient, err := triton.NewClient(
		"localhost:8000", // Triton HTTP endpoint
		false,            // Verbose logging
		3,                // Connection timeout in seconds
		3,                // Network timeout in seconds
		false,            // Use SSL
		false,            // Insecure connection
		nil,              // Custom HTTP client (optional)
		nil,              // Logger (optional)
	)
	if err != nil {
		log.Fatalf("Failed to create Triton HTTP client: %v", err)
	}

	// Sample text to encode
	text := "Hello, Triton Inference Server!"

	// Encoding options
	encodeOpts := &tokenizerOptions.EncodeOptions{
		ReturnAllAttributes: true,
	}

	// ---------------------------
	// Inference with tyBERT Model
	// ---------------------------

	// Initialize the tokenizer for tyBERT
	tybertTokenizer := tokenizer.NewTokenizer("ty_bert/tokenizer.json")

	// Encode the text using the tyBERT tokenizer
	tybertEncodeResp := tybertTokenizer.Encode(text, encodeOpts)

	// Display encoding results
	fmt.Printf("tyBERT Encoded IDs: %v\n", tybertEncodeResp.IDs)
	fmt.Printf("tyBERT Tokens: %v\n", tybertEncodeResp.Tokens)
	fmt.Printf("tyBERT Offsets: %v\n", tybertEncodeResp.Offsets)
	fmt.Printf("tyBERT Type IDs: %v\n", tybertEncodeResp.TypeIDs)
	fmt.Printf("tyBERT Special Tokens Mask: %v\n", tybertEncodeResp.SpecialTokensMask)
	fmt.Printf("tyBERT Attention Mask: %v\n", tybertEncodeResp.AttentionMask)

	// Prepare inference inputs for tyBERT
	sequenceLength := int64(len(tybertEncodeResp.IDs))
	batchSize := int64(1) // Since we're processing a single sequence

	tybertInputIDs := triton.NewInferInput("input_ids", "INT64", []int64{batchSize, sequenceLength}, nil)
	err = tybertInputIDs.SetData(convertUint32ToInt64(tybertEncodeResp.IDs), true)
	if err != nil {
		log.Fatalf("Failed to set input_ids data for tyBERT: %v", err)
	}

	tybertTokenTypeIDs := triton.NewInferInput("token_type_ids", "INT64", []int64{batchSize, sequenceLength}, nil)
	err = tybertTokenTypeIDs.SetData(convertUint32ToInt64(tybertEncodeResp.TypeIDs), true)
	if err != nil {
		log.Fatalf("Failed to set token_type_ids data for tyBERT: %v", err)
	}

	tybertAttentionMask := triton.NewInferInput("attention_mask", "INT64", []int64{batchSize, sequenceLength}, nil)
	err = tybertAttentionMask.SetData(convertUint32ToInt64(tybertEncodeResp.AttentionMask), true)
	if err != nil {
		log.Fatalf("Failed to set attention_mask data for tyBERT: %v", err)
	}

	// Define the desired output tensors for tyBERT
	tybertOutputs := []base.InferOutput{
		triton.NewInferOutput("logits", map[string]any{"binary_data": true}),
	}

	// Perform inference using the tyBERT model
	tybertResponse, err := tritonClient.Infer(
		context.Background(),
		"ty_bert", // Model name
		"1",       // Model version
		[]base.InferInput{tybertInputIDs, tybertTokenTypeIDs, tybertAttentionMask},
		tybertOutputs,
		nil, // Additional options (optional)
	)
	if err != nil {
		log.Fatalf("Failed to perform inference on tyBERT: %v", err)
	}

	// Extract the logits from the response
	tybertLogitsData, err := tybertResponse.AsFloat16Slice("logits")
	if err != nil {
		log.Fatalf("Failed to extract logits from tyBERT response: %v", err)
	}
	shape, err := tybertResponse.GetShape("logits")
	if err != nil {
		log.Fatalf("Failed to extract shape from response: %v", err)
	}

	// Reshape the logits
	tybertReshapedLogits, err := converter.Reshape3D[float64](tybertLogitsData, shape)
	if err != nil {
		log.Fatal("Failed to parse inference response")
	}

	// Post-process the logits (e.g., mean pooling)
	postProcessMgr := postprocess.NewPostprocessManager()
	tybertMeanPooling, err := postProcessMgr.MeanPoolingFloat64Slice3D(
		tybertReshapedLogits,
		[][]int64{convertUint32ToInt64(tybertEncodeResp.AttentionMask)},
	)
	if err != nil {
		log.Fatalf("Failed to perform mean pooling on tyBERT logits: %v", err)
	}

	fmt.Printf("tyBERT Inference Mean Pooling Result: %v\n", tybertMeanPooling)

	// ------------------------------
	// Inference with tyRoBERTa Model
	// ------------------------------

	// Initialize the tokenizer for tyRoBERTa
	tyrobertaTokenizer := tokenizer.NewTokenizer("/tokenizers/ty_roberta/tokenizer.json")

	// Encode the text using the tyRoBERTa tokenizer
	tyrobertaEncodeResp := tyrobertaTokenizer.Encode(text, encodeOpts)

	// Display encoding results
	fmt.Printf("tyRoBERTa Encoded IDs: %v\n", tyrobertaEncodeResp.IDs)
	fmt.Printf("tyRoBERTa Tokens: %v\n", tyrobertaEncodeResp.Tokens)
	fmt.Printf("tyRoBERTa Offsets: %v\n", tyrobertaEncodeResp.Offsets)
	fmt.Printf("tyRoBERTa Type IDs: %v\n", tyrobertaEncodeResp.TypeIDs)
	fmt.Printf("tyRoBERTa Special Tokens Mask: %v\n", tyrobertaEncodeResp.SpecialTokensMask)
	fmt.Printf("tyRoBERTa Attention Mask: %v\n", tyrobertaEncodeResp.AttentionMask)

	// Prepare inference inputs for tyRoBERTa
	sequenceLength = int64(len(tyrobertaEncodeResp.IDs))

	tyrobertaInputIDs := triton.NewInferInput("input_ids", "INT64", []int64{batchSize, sequenceLength}, nil)
	err = tyrobertaInputIDs.SetData(convertUint32ToInt64(tyrobertaEncodeResp.IDs), true)
	if err != nil {
		log.Fatalf("Failed to set input_ids data for tyRoBERTa: %v", err)
	}

	tyrobertaAttentionMask := triton.NewInferInput("attention_mask", "INT64", []int64{batchSize, sequenceLength}, nil)
	err = tyrobertaAttentionMask.SetData(convertUint32ToInt64(tyrobertaEncodeResp.AttentionMask), true)
	if err != nil {
		log.Fatalf("Failed to set attention_mask data for tyRoBERTa: %v", err)
	}

	// Define the desired output tensors for tyRoBERTa
	tyrobertaOutputs := []base.InferOutput{
		triton.NewInferOutput("logits", map[string]any{"binary_data": true}),
	}

	// Perform inference using the tyRoBERTa model
	tyrobertaResponse, err := tritonClient.Infer(
		context.Background(),
		"ty_roberta", // Model name
		"1",          // Model version
		[]base.InferInput{tyrobertaInputIDs, tyrobertaAttentionMask},
		tyrobertaOutputs,
		nil, // Additional options (optional)
	)
	if err != nil {
		log.Fatalf("Failed to perform inference on tyRoBERTa: %v", err)
	}

	// Extract the logits from the response
	tyrobertaLogitsData, err := tyrobertaResponse.AsFloat16Slice("logits")
	if err != nil {
		log.Fatalf("Failed to extract logits from tyRoBERTa response: %v", err)
	}

	// Reshape the logits
	tyrobertaReshapedLogits, err := converter.Reshape3D[float64](tyrobertaLogitsData, shape)
	if err != nil {
		log.Fatal("Failed to parse inference response")
	}

	// Post-process the logits (e.g., mean pooling)
	tyrobertaMeanPooling, err := postProcessMgr.MeanPoolingFloat64Slice3D(
		tyrobertaReshapedLogits,
		[][]int64{convertUint32ToInt64(tyrobertaEncodeResp.AttentionMask)},
	)
	if err != nil {
		log.Fatalf("Failed to perform mean pooling on tyRoBERTa logits: %v", err)
	}

	fmt.Printf("tyRoBERTa Inference Mean Pooling Result: %v\n", tyrobertaMeanPooling)
}

// convertUint32ToInt64 converts a slice of uint32 to int64
func convertUint32ToInt64(input []uint32) []int64 {
	result := make([]int64, len(input))
	for i, v := range input {
		result[i] = int64(v)
	}
	return result
}
