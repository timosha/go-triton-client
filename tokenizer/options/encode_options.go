package options

type EncodeOptions struct {
	ReturnAllAttributes     bool
	ReturnAttentionMask     bool
	EncodeSpecialTokens     bool
	ReturnTokens            bool
	ReturnOffsets           bool
	ReturnSpecialTokensMask bool
	ReturnTypeIDs           bool
}
