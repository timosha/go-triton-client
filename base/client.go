package base

import (
	"context"
	"github.com/Trendyol/go-triton-client/models"
	"github.com/Trendyol/go-triton-client/options"
)

type Client interface {
	IsServerLive(ctx context.Context, options *options.Options) (bool, error)
	IsServerReady(ctx context.Context, options *options.Options) (bool, error)
	IsModelReady(ctx context.Context, modelName string, modelVersion string, options *options.Options) (bool, error)
	GetServerMetadata(ctx context.Context, options *options.Options) (*models.ServerMetadataResponse, error)
	GetModelMetadata(ctx context.Context, modelName string, modelVersion string, options *options.Options) (*models.ModelMetadataResponse, error)
	GetModelConfig(ctx context.Context, modelName string, modelVersion string, options *options.Options) (*models.ModelConfigResponse, error)
	GetModelRepositoryIndex(ctx context.Context, options *options.Options) ([]models.ModelRepositoryIndexResponse, error)
	LoadModel(ctx context.Context, modelName string, config string, files map[string][]byte, options *options.Options) error
	UnloadModel(ctx context.Context, modelName string, unloadDependents bool, options *options.Options) error
	GetInferenceStatistics(ctx context.Context, modelName string, modelVersion string, options *options.Options) (*models.InferenceStatisticsResponse, error)
	GetTraceSettings(ctx context.Context, modelName string, options *options.Options) (*models.TraceSettingsResponse, error)
	UpdateLogSettings(ctx context.Context, request models.LogSettingsRequest, options *options.Options) error
	GetLogSettings(ctx context.Context, options *options.Options) (*models.LogSettingsResponse, error)
	GetSystemSharedMemoryStatus(ctx context.Context, regionName string, options *options.Options) ([]models.SystemSharedMemoryStatusResponse, error)
	RegisterSystemSharedMemory(ctx context.Context, name string, key string, byteSize int, offset int, options *options.Options) error
	UnregisterSystemSharedMemory(ctx context.Context, name string, options *options.Options) error
	GetCUDASharedMemoryStatus(ctx context.Context, regionName string, options *options.Options) ([]models.CUDASharedMemoryStatusResponse, error)
	RegisterCUDASharedMemory(ctx context.Context, name string, rawHandle []byte, deviceID int, byteSize int, options *options.Options) error
	UnregisterCUDASharedMemory(ctx context.Context, name string, options *options.Options) error
	Infer(
		ctx context.Context,
		modelName string,
		modelVersion string,
		inputs []InferInput,
		outputs []InferOutput,
		options *options.InferOptions,
	) (InferResult, error)
}
