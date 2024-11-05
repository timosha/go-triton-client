package tokenizer

import (
	"github.com/Trendyol/go-triton-client/tokenizer/models"
	"github.com/Trendyol/go-triton-client/tokenizer/options"
	"github.com/daulet/tokenizers"
	"log"
	"strings"
)

/*
// macOS (darwin)
#cgo darwin,amd64 LDFLAGS: -L${SRCDIR}/lib/darwin/amd64 -ltokenizers
#cgo darwin,arm64 LDFLAGS: -L${SRCDIR}/lib/darwin/arm64 -ltokenizers

// Linux
#cgo linux,amd64 LDFLAGS: -L${SRCDIR}/lib/linux/amd64 -ltokenizers
#cgo linux,arm64 LDFLAGS: -L${SRCDIR}/lib/linux/arm64 -ltokenizers
*/
import "C"

type Tokenizer interface {
	Encode(text string, encodeOptions *options.EncodeOptions) *models.EncodeResponse
	Decode(tokenIDs []uint32, decodeOptions *options.DecodeOptions) *models.DecodeResponse
}

type tokenizer struct {
	tk *tokenizers.Tokenizer
}

func NewTokenizer(path string) Tokenizer {
	if strings.TrimSpace(path) == "" {
		log.Fatal("path can't be empty")
	}

	tk, err := tokenizers.FromFile(path)
	if err != nil {
		log.Fatal(err)
	}
	return &tokenizer{tk: tk}
}

func (t *tokenizer) Encode(text string, encodeOptions *options.EncodeOptions) *models.EncodeResponse {
	var options []tokenizers.EncodeOption
	encodeSpecialTokens := false

	if encodeOptions == nil {
		options = append(options, tokenizers.WithReturnTokens())
	} else {
		if encodeOptions.ReturnAllAttributes {
			options = append(options, tokenizers.WithReturnAllAttributes())
		}

		if encodeOptions.ReturnAttentionMask {
			options = append(options, tokenizers.WithReturnAttentionMask())
		}

		if encodeOptions.ReturnTokens {
			options = append(options, tokenizers.WithReturnTokens())
		}

		if encodeOptions.ReturnOffsets {
			options = append(options, tokenizers.WithReturnOffsets())
		}

		if encodeOptions.ReturnSpecialTokensMask {
			options = append(options, tokenizers.WithReturnSpecialTokensMask())
		}

		if encodeOptions.ReturnTypeIDs {
			options = append(options, tokenizers.WithReturnTypeIDs())
		}

		if encodeOptions.EncodeSpecialTokens {
			encodeSpecialTokens = true
		}
	}

	encoded := t.tk.EncodeWithOptions(text, encodeSpecialTokens, options...)
	offsets := make([]models.Offset, len(encoded.Offsets))
	for i, extOffset := range encoded.Offsets {
		offsets[i] = models.Offset{extOffset[0], extOffset[1]}
	}

	encodeResponse := &models.EncodeResponse{
		IDs:               encoded.IDs,
		TypeIDs:           encoded.TypeIDs,
		SpecialTokensMask: encoded.SpecialTokensMask,
		AttentionMask:     encoded.AttentionMask,
		Tokens:            encoded.Tokens,
		Offsets:           offsets,
	}
	return encodeResponse
}

func (t *tokenizer) Decode(tokenIDs []uint32, decodeOptions *options.DecodeOptions) *models.DecodeResponse {
	skipSpecialTokens := false
	if decodeOptions != nil {
		skipSpecialTokens = decodeOptions.SkipSpecialTokens
	}

	decoded := t.tk.Decode(tokenIDs, skipSpecialTokens)

	decodeResponse := &models.DecodeResponse{
		Decoded: decoded,
	}
	return decodeResponse
}
