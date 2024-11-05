package models

type TraceSettingsResponse struct {
	TraceLevel   []string `json:"trace_level"`
	TraceRate    string   `json:"trace_rate"`
	TraceCount   string   `json:"trace_count"`
	LogFrequency string   `json:"log_frequency"`
	TraceFile    string   `json:"trace_file"`
}
