package base

// Marshaller interface defines a method for marshaling data into byte format.
type Marshaller interface {
	Marshal(v any) ([]byte, error)
}
