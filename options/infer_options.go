package options

type InferOptions struct {
	Headers                      map[string]string
	QueryParams                  map[string]string
	RequestID                    *string
	SequenceID                   *int
	SequenceStart                *bool
	SequenceEnd                  *bool
	Priority                     *int
	Timeout                      *int
	RequestCompressionAlgorithm  *string
	ResponseCompressionAlgorithm *string
	Parameters                   map[string]any
}
