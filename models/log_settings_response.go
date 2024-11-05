package models

type LogSettingsResponse struct {
	LogFile         string `json:"log_file"`
	LogInfo         bool   `json:"log_info"`
	LogWarning      bool   `json:"log_warning"`
	LogError        bool   `json:"log_error"`
	LogVerboseLevel int    `json:"log_verbose_level"`
	LogFormat       string `json:"log_format"`
}
