package parser

import (
	"reflect"
)

// ParseSlice attempts to convert an interface{} containing data into a specified slice type S.
// It handles nested slice structures and performs type conversion as needed. The function
// returns the converted slice of type S and a boolean indicating whether the conversion was successful.
func ParseSlice[S any](data interface{}) (S, bool) {
	if data == nil {
		var empty S
		return empty, false
	}

	// Start the conversion process from the top-level slice
	v := reflect.ValueOf(data)
	targetType := reflect.TypeOf((*S)(nil)).Elem()

	// Directly return if types match
	if v.Type().ConvertibleTo(targetType) {
		return v.Convert(targetType).Interface().(S), true
	}

	// Recursive function to handle different slice shapes
	var convert func(reflect.Value, reflect.Type) (reflect.Value, bool)
	convert = func(v reflect.Value, t reflect.Type) (reflect.Value, bool) {
		// Handle the case where the value is an interface
		if v.Kind() == reflect.Interface {
			v = v.Elem()
		}

		// If the target type is not a slice, attempt direct conversion
		if t.Kind() != reflect.Slice {
			if v.CanConvert(t) {
				return v.Convert(t), true
			}
			return reflect.Value{}, false
		}

		// If it is a slice, get the element type and recursively convert each element
		if v.Kind() != reflect.Slice {
			return reflect.Value{}, false
		}

		elemType := t.Elem()
		result := reflect.MakeSlice(reflect.SliceOf(elemType), v.Len(), v.Len())
		for i := 0; i < v.Len(); i++ {
			converted, ok := convert(v.Index(i), elemType)
			if !ok {
				return reflect.Value{}, false
			}
			result.Index(i).Set(converted)
		}
		return result, true
	}

	// If direct conversion is not possible, try recursive conversion
	convertedValue, ok := convert(v, targetType)
	if !ok {
		var empty S
		return empty, false
	}

	return convertedValue.Interface().(S), true
}
