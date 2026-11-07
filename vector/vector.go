// Package vector is the typed input shape for dense rows: one flat,
// row-major buffer of a supported scalar type and its width. Float32Rows
// turns it into the [][]float32 rows kmeans, rabitq and the root package's
// float32 entry points take; Rows and Row read it a row at a time, which
// is how the typed encoders (ivfq.DenseEncoder, rabitq.BlockEncoder) avoid
// a float32 copy of the whole input.
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
	if _, err := s.Rows(); err != nil {
		return nil, err
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
		flat = widenInto(make([]float32, len(d)), d)
	case []uint8:
		flat = widenInto(make([]float32, len(d)), d)
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

// Rows is the number of rows in s, or an error when Dims is not positive or
// Data does not hold whole rows. It reads no values.
func (s Slice[T]) Rows() (int, error) {
	if s.Dims <= 0 {
		return 0, fmt.Errorf("vector: dims must be positive, got %d", s.Dims)
	}
	if len(s.Data)%s.Dims != 0 {
		return 0, fmt.Errorf("vector: %d values are not whole %d-wide rows", len(s.Data), s.Dims)
	}
	return len(s.Data) / s.Dims, nil
}

// Row returns row i as the float32 values Float32Rows gives it, one row
// at a time: for float32 a read-only view of Data whose capacity is its
// length (dst is unused), for other types row i decoded into dst[:Dims],
// which it returns. It does not check that values are finite. i must be a
// row of s and, unless T is float32, dst at least Dims long.
func (s Slice[T]) Row(i int, dst []float32) []float32 {
	lo, hi := i*s.Dims, (i+1)*s.Dims
	switch d := any(s.Data).(type) {
	case []float32:
		return d[lo:hi:hi]
	case []float16.Float16:
		dst = dst[:s.Dims]
		for j, h := range d[lo:hi] {
			dst[j] = h.Float32()
		}
		return dst
	case []int8:
		return widenInto(dst[:s.Dims], d[lo:hi])
	case []uint8:
		return widenInto(dst[:s.Dims], d[lo:hi])
	}
	panic("unreachable")
}

func widenInto[T int8 | uint8](dst []float32, d []T) []float32 {
	for j, x := range d {
		dst[j] = float32(x)
	}
	return dst
}
