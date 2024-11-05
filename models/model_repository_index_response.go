package models

type ModelRepositoryIndexResponse struct {
	Name    string `json:"name"`
	Version string `json:"version"`
	State   string `json:"state"`
	Reason  string `json:"reason"`
}
