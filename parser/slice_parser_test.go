package parser

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestParseSlice(t *testing.T) {
	data1 := []int{1, 2, 3}
	expected1 := []int{1, 2, 3}
	result1, ok1 := ParseSlice[[]int](data1)
	assert.True(t, ok1)
	assert.Equal(t, expected1, result1)

	data2 := []float64{1.1, 2.2, 3.3}
	expected2 := []float64{1.1, 2.2, 3.3}
	result2, ok2 := ParseSlice[[]float64](data2)
	assert.True(t, ok2)
	assert.Equal(t, expected2, result2)

	data3 := []int{1, 2, 3}
	expected3 := []float64{1.0, 2.0, 3.0}
	result3, ok3 := ParseSlice[[]float64](data3)
	assert.True(t, ok3)
	assert.Equal(t, expected3, result3)

	data4 := [][]int{{1, 2}, {3, 4}}
	expected4 := [][]int{{1, 2}, {3, 4}}
	result4, ok4 := ParseSlice[[][]int](data4)
	assert.True(t, ok4)
	assert.Equal(t, expected4, result4)

	data5 := []int{1, 2, 3}
	expected5 := []string{"\x01", "\x02", "\x03"}
	result5, ok5 := ParseSlice[[]string](data5)
	assert.True(t, ok5)
	assert.Equal(t, expected5, result5)

	var data6 interface{}
	var expected6 []int
	result6, ok6 := ParseSlice[[]int](data6)
	assert.False(t, ok6)
	assert.Equal(t, expected6, result6)

	data7 := [][]int{{1, 2}, {3, 4}}
	expected7 := [][]float64{{1.0, 2.0}, {3.0, 4.0}}
	result7, ok7 := ParseSlice[[][]float64](data7)
	assert.True(t, ok7)
	assert.Equal(t, expected7, result7)

	data8 := 42
	expected8 := 42
	result8, ok8 := ParseSlice[int](data8)
	assert.True(t, ok8)
	assert.Equal(t, expected8, result8)

	data9 := "test"
	expected9 := "test"
	result9, ok9 := ParseSlice[string](data9)
	assert.True(t, ok9)
	assert.Equal(t, expected9, result9)

	data10 := []interface{}{1, 2.0, "three"}
	expected10 := []interface{}{1, 2.0, "three"}
	result10, ok10 := ParseSlice[[]interface{}](data10)
	assert.True(t, ok10)
	assert.Equal(t, expected10, result10)

	data11 := []interface{}{1, 2, 3}
	expected11 := []float64{1.0, 2.0, 3.0}
	result11, ok11 := ParseSlice[[]float64](data11)
	assert.True(t, ok11)
	assert.Equal(t, expected11, result11)

	data12 := [][]interface{}{{1, 2}, {3, 4.0}}
	expected12 := [][]float64{{1.0, 2.0}, {3.0, 4.0}}
	result12, ok12 := ParseSlice[[][]float64](data12)
	assert.True(t, ok12)
	assert.Equal(t, expected12, result12)

	data13 := []struct {
		Name string
		Age  int
	}{{"Alice", 30}, {"Bob", 25}}

	var expected13 []string
	result13, ok13 := ParseSlice[[]string](data13)
	assert.False(t, ok13)
	assert.Equal(t, expected13, result13)

	data14 := 42
	var expected14 []int
	result14, ok14 := ParseSlice[[]int](data14)
	assert.False(t, ok14)
	assert.Equal(t, expected14, result14)
}
