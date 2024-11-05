package grpc

import (
	"github.com/Trendyol/go-triton-client/client/grpc/grpc_generated_v2"
	"github.com/Trendyol/go-triton-client/models"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestMapInferenceStatWithNil(t *testing.T) {
	result := mapInferenceStat(nil)
	expected := models.InferenceStatisticsStat{}
	assert.Equal(t, expected, result)
}

func TestMapInferenceStatWithValidStat(t *testing.T) {
	stat := &grpc_generated_v2.StatisticDuration{
		Count: 5,
		Ns:    1000,
	}

	result := mapInferenceStat(stat)
	expected := models.InferenceStatisticsStat{
		Count:       5,
		Nanoseconds: 1000,
	}
	assert.Equal(t, expected, result)
}

func TestMapBatchStatsWithEmptySlice(t *testing.T) {
	result := mapBatchStats([]*grpc_generated_v2.InferBatchStatistics{})
	assert.Empty(t, result)
}

func TestMapBatchStatsWithValidData(t *testing.T) {
	batchStats := []*grpc_generated_v2.InferBatchStatistics{
		{BatchSize: 1},
		{BatchSize: 2},
	}

	result := mapBatchStats(batchStats)
	assert.Equal(t, len(batchStats), len(result))
	assert.Equal(t, batchStats[0], result[0])
	assert.Equal(t, batchStats[1], result[1])
}

func TestMapMemoryUsageWithEmptySlice(t *testing.T) {
	result := mapMemoryUsage([]*grpc_generated_v2.MemoryUsage{})
	assert.Empty(t, result)
}

func TestMapMemoryUsageWithValidData(t *testing.T) {
	memoryUsage := []*grpc_generated_v2.MemoryUsage{
		{ByteSize: 1024},
		{ByteSize: 2048},
	}

	result := mapMemoryUsage(memoryUsage)
	assert.Equal(t, len(memoryUsage), len(result))
	assert.Equal(t, memoryUsage[0], result[0])
	assert.Equal(t, memoryUsage[1], result[1])
}
