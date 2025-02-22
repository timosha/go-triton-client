package models

type ModelConfigResponse struct {
	Name                 string                               `json:"name"`
	Platform             string                               `json:"platform"`
	Backend              string                               `json:"backend"`
	VersionPolicy        ModelConfigVersionPolicy             `json:"version_policy"`
	Input                []ModelConfigInput                   `json:"input"`
	Output               []ModelConfigOutput                  `json:"output"`
	BatchInput           []any                                `json:"batch_input"`
	BatchOutput          []any                                `json:"batch_output"`
	Optimization         ModelConfigOptimization              `json:"optimization"`
	InstanceGroup        []ModelConfigInstanceGroup           `json:"instance_group"`
	DefaultModelFileName string                               `json:"default_model_file_name"`
	CCModelFileNames     any                                  `json:"cc_model_file_names"`
	MetricTags           map[string]any                       `json:"metric_tags"`
	Parameters           map[string]ModelConfigParameterValue `json:"parameters"`
	ModelWarmup          []any                                `json:"model_warmup"`
}

type ModelConfigVersionPolicy struct {
	Latest ModelConfigLatestVersionPolicy `json:"latest"`
}

type ModelConfigLatestVersionPolicy struct {
	NumVersions int `json:"num_versions"`
}

type ModelConfigInput struct {
	Name             string `json:"name"`
	DataType         string `json:"data_type"`
	Format           string `json:"format"`
	Dims             []int  `json:"dims"`
	IsShapeTensor    bool   `json:"is_shape_tensor"`
	AllowRaggedBatch bool   `json:"allow_ragged_batch"`
	Optional         bool   `json:"optional"`
}

type ModelConfigOutput struct {
	Name          string `json:"name"`
	DataType      string `json:"data_type"`
	Format        string `json:"format"`
	Dims          []int  `json:"dims"`
	LabelFilename string `json:"label_filename"`
	IsShapeTensor bool   `json:"is_shape_tensor"`
}

type ModelConfigOptimization struct {
	Priority                    string                        `json:"priority"`
	InputPinnedMemory           ModelConfigInputPinnedMemory  `json:"input_pinned_memory"`
	OutputPinnedMemory          ModelConfigOutputPinnedMemory `json:"output_pinned_memory"`
	GatherKernelBufferThreshold int                           `json:"gather_kernel_buffer_threshold"`
	EagerBatching               bool                          `json:"eager_batching"`
}

type ModelConfigInputPinnedMemory struct {
	Enable bool `json:"enable"`
}

type ModelConfigOutputPinnedMemory struct {
	Enable bool `json:"enable"`
}

type ModelConfigInstanceGroup struct {
	Name             string   `json:"name"`
	Kind             string   `json:"kind"`
	Count            int      `json:"count"`
	GPUs             []string `json:"gpus"`
	SecondaryDevices []string `json:"secondary_devices"`
	Profile          []any    `json:"profile"`
	Passive          bool     `json:"passive"`
	HostPolicy       string   `json:"host_policy"`
}

type ModelConfigParameterValue struct {
	StringValue string `json:"string_value"`
}
