package models

type ServerMetadataResponse struct {
	Name       string   `json:"name"`
	Version    string   `json:"version"`
	Extensions []string `json:"extensions"`
}
