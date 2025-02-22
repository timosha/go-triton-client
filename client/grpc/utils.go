package grpc

import (
	"github.com/Trendyol/go-triton-client/client/grpc/grpc_generated_v2"
	"github.com/Trendyol/go-triton-client/models"
)

// mapInferenceStat maps gRPC StatisticDuration to a local InferenceStatisticsStat model.
func mapInferenceStat(stat *grpc_generated_v2.StatisticDuration) models.InferenceStatisticsStat {
	if stat == nil {
		return models.InferenceStatisticsStat{}
	}

	return models.InferenceStatisticsStat{
		Count:       int(stat.Count),
		Nanoseconds: int(stat.Ns),
	}
}

// mapBatchStats maps a slice of InferBatchStatistics into a slice of empty interfaces.
func mapBatchStats(batchStats []*grpc_generated_v2.InferBatchStatistics) []any {
	result := make([]any, len(batchStats))
	for i, bs := range batchStats {
		result[i] = bs
	}
	return result
}

// mapMemoryUsage maps a slice of MemoryUsage into a slice of empty interfaces.
func mapMemoryUsage(memoryUsage []*grpc_generated_v2.MemoryUsage) []any {
	result := make([]any, len(memoryUsage))
	for i, mu := range memoryUsage {
		result[i] = mu
	}
	return result
}
