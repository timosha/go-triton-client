package models

type EncodeResponse struct {
	IDs               []uint32 `json:"ids"`
	TypeIDs           []uint32 `json:"type_ids"`
	SpecialTokensMask []uint32 `json:"special_tokens_mask"`
	AttentionMask     []uint32 `json:"attention_mask"`
	Tokens            []string `json:"tokens"`
	Offsets           []Offset `json:"offsets"`
}

type Offset [2]uint
