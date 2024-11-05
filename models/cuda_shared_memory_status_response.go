package models

type CUDASharedMemoryStatusResponse struct {
	Name     string `json:"name"`
	ByteSize uint64 `json:"byte_size"`
	DeviceId uint64 `json:"device_id"`
}
