package base

// RequestWrapper defines an interface for preparing inference requests.
type RequestWrapper interface {
	PrepareRequest() (interface{}, error)
}
