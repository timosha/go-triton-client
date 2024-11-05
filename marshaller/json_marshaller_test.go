package marshaller

import (
	"encoding/json"
	"reflect"
	"testing"
)

func TestJSONMarshaller_Marshal(t *testing.T) {
	marshaller := NewJSONMarshaller()
	data := "test-data"
	marshalled1, _ := json.Marshal(data)
	marshalled2, _ := marshaller.Marshal(data)
	if !reflect.DeepEqual(marshalled1, marshalled2) {
		t.Errorf("expecting %v got %v", marshalled1, marshalled2)
	}
}
