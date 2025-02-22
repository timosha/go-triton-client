package models

type InferenceStatisticsResponse struct {
	ModelStats []InferenceStatisticsModelStat `json:"model_stats"`
}

type InferenceStatisticsModelStat struct {
	Name           string                            `json:"name"`
	Version        string                            `json:"version"`
	LastInference  int                               `json:"last_inference"`
	InferenceCount int                               `json:"inference_count"`
	ExecutionCount int                               `json:"execution_count"`
	InferenceStats InferenceStatisticsInferenceStats `json:"inference_stats"`
	BatchStats     []any                             `json:"batch_stats"`
	MemoryUsage    []any                             `json:"memory_usage"`
}

type InferenceStatisticsInferenceStats struct {
	Success       InferenceStatisticsStat `json:"success"`
	Fail          InferenceStatisticsStat `json:"fail"`
	Queue         InferenceStatisticsStat `json:"queue"`
	ComputeInput  InferenceStatisticsStat `json:"compute_input"`
	ComputeInfer  InferenceStatisticsStat `json:"compute_infer"`
	ComputeOutput InferenceStatisticsStat `json:"compute_output"`
	CacheHit      InferenceStatisticsStat `json:"cache_hit"`
	CacheMiss     InferenceStatisticsStat `json:"cache_miss"`
}

type InferenceStatisticsStat struct {
	Count       int `json:"count"`
	Nanoseconds int `json:"ns"`
}
