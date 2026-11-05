// Package vector is the typed input shape for dense rows: one flat,
// row-major buffer of a supported scalar type and its width. Float32Rows
// turns it into the [][]float32 rows kmeans, rabitq and the root package
// take, so those entry points stay float32.
//
// It imports nothing from this module, so every other package may use it.
package vector

import (
	"fmt"
	"math"

	"github.com/x448/float16"
)

// Scalar is a coordinate type Slice can hold.
type Scalar interface {
	float32 | float16.Float16 | int8 | uint8
}

// Slice is len(Data)/Dims rows of Dims coordinates each, row-major.
type Slice[T Scalar] struct {
	Dims int
	Data []T
}

// Float32Rows returns s as float32 rows. Dims must be positive, Data must
// hold whole rows, and every value must be finite. An empty Data is zero
// rows. Half values are decoded as IEEE binary16, not converted as integers.
//
// The rows are read-only and may alias Data: float32 rows are views of Data
// itself, and other types are decoded into one new buffer the rows share.
// Do not write through the rows, and do not change Data while the rows are
// in use. Float32Rows does not modify Data. Each row's capacity is its
// length, so appending to a row copies it instead of overwriting the next.
func (s Slice[T]) Float32Rows() ([][]float32, error) {
	if s.Dims <= 0 {
		return nil, fmt.Errorf("vector: dims must be positive, got %d", s.Dims)
	}
	if len(s.Data)%s.Dims != 0 {
		return nil, fmt.Errorf("vector: %d values are not whole %d-wide rows", len(s.Data), s.Dims)
	}
	var flat []float32
	switch d := any(s.Data).(type) {
	case []float32:
		flat = d
	case []float16.Float16:
		flat = make([]float32, len(d))
		for i, h := range d {
			flat[i] = h.Float32()
		}
	case []int8:
		flat = widen(d)
	case []uint8:
		flat = widen(d)
	}
	for i, x := range flat {
		if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
			return nil, fmt.Errorf("vector: row %d coordinate %d is %v", i/s.Dims, i%s.Dims, x)
		}
	}
	rows := make([][]float32, len(flat)/s.Dims)
	for i := range rows {
		rows[i] = flat[i*s.Dims : (i+1)*s.Dims : (i+1)*s.Dims]
	}
	return rows, nil
}

func widen[T int8 | uint8](d []T) []float32 {
	out := make([]float32, len(d))
	for i, x := range d {
		out[i] = float32(x)
	}
	return out
}
