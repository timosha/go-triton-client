package base

// Marshaller interface defines a method for marshaling data into byte format.
type Marshaller interface {
	Marshal(v interface{}) ([]byte, error)
}
