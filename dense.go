package ivfq

import (
	"errors"
	"fmt"
	"math"

	"github.com/axiomhq/ivfq/vector"
	"github.com/x448/float16"
)

// Dense is a CentroidBlock of rows stored as T and decoded to float32 to
// score. NewDense[T] is its encoder for BuildTree.
//
// float32 rows are kept exactly. float16 rows are rounded to the nearest
// IEEE binary16 value; a coordinate that rounds past ±65504 (|x| ≥ 65520)
// is an error, not an infinity. int8 and uint8 rows are codes on one affine scale per
// block, because tree means are fractional: the block's [min, max]
// coordinate range maps onto all 256 codes, scale = (max-min)/255 and
// offset = min - scale*(T's minimum), a coordinate x is stored as
// round((x-offset)/scale), halves away from zero, clamped to T's range,
// and decodes to offset + scale*code, within scale/2 of x up to float32
// rounding. A block whose coordinates are all equal decodes them exactly.
//
// ScoreAll scores the decoded rows exactly as a Dense[float32] of them
// would: -(‖q‖² + ‖row‖² - 2·q·row), combined in float64, FloatTree's
// ranking distance negated.
type Dense[T vector.Scalar] struct {
	rows          vector.Slice[T]
	norms         []float32 // squared norm of each decoded row
	scale, offset float32   // integer T: decoded = offset + scale*code
}

// NewDense encodes rows, which must be nonempty, equally wide, at least
// one-dimensional and finite, and each row as encoded must have a finite
// float32 squared norm (FloatTree's rule: no coordinate past about 1.8e19).
// It copies them and keeps no float32 copy unless T is float32.
func NewDense[T vector.Scalar](rows [][]float32) (Dense[T], error) {
	if len(rows) == 0 || len(rows[0]) == 0 {
		return Dense[T]{}, errEmptyDense
	}
	for i, r := range rows {
		if len(r) != len(rows[0]) {
			return Dense[T]{}, fmt.Errorf("ivfq: row %d is %d-dimensional, want %d", i, len(r), len(rows[0]))
		}
	}
	return newDense[T](len(rows), len(rows[0]), func(i int, _ []float32) []float32 { return rows[i] })
}

// DenseEncoder is the Encoder from rows of S to Dense[T] blocks. Encode
// reads S's values as float32 (float16 decoded as IEEE binary16), then
// encodes them as NewDense[T] does, a row at a time: it borrows rows only
// for the call, keeps nothing of them, and never builds a float32 copy of
// the whole input.
type DenseEncoder[S, T vector.Scalar] struct{}

// Encode implements Encoder.
func (DenseEncoder[S, T]) Encode(rows vector.Slice[S]) (Dense[T], error) {
	n, err := rows.Rows()
	if err != nil {
		return Dense[T]{}, err
	}
	return newDense[T](n, rows.Dims, rows.Row)
}

var errEmptyDense = errors.New("ivfq: Dense needs at least one row of at least one dimension")

// newDense is NewDense over n rows of dims values, row i read by row, which
// may decode into the dims-wide scratch it is given.
func newDense[T vector.Scalar](n, dims int, row func(i int, dst []float32) []float32) (Dense[T], error) {
	if n == 0 || dims == 0 {
		return Dense[T]{}, errEmptyDense
	}
	scratch := make([]float32, dims)
	lo, hi := math.Inf(1), math.Inf(-1)
	for i := range n {
		for j, x := range row(i, scratch) {
			if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
				return Dense[T]{}, fmt.Errorf("ivfq: row %d coordinate %d is %v", i, j, x)
			}
			lo, hi = min(lo, float64(x)), max(hi, float64(x))
		}
	}
	d := Dense[T]{rows: vector.Slice[T]{Dims: dims, Data: make([]T, n*dims)}, norms: make([]float32, n), scale: 1}
	switch any(d.rows.Data).(type) {
	case []int8:
		d.scale, d.offset = affine(lo, hi, -128, 127)
	case []uint8:
		d.scale, d.offset = affine(lo, hi, 0, 255)
	}
	for i := range n {
		src := row(i, scratch)
		switch dst := any(d.rows.Data[i*dims : (i+1)*dims]).(type) {
		case []float32:
			copy(dst, src)
		case []float16.Float16:
			for j, x := range src {
				h := float16.Fromfloat32(x)
				if !h.IsFinite() {
					return Dense[T]{}, fmt.Errorf("ivfq: row %d coordinate %d is %v, past float16's range", i, j, x)
				}
				dst[j] = h
			}
		case []int8:
			encodeCodes(dst, src, d.scale, d.offset, -128, 127)
		case []uint8:
			encodeCodes(dst, src, d.scale, d.offset, 0, 255)
		}
		norm, ok := finiteNorm(d.row(i, scratch))
		if !ok {
			return Dense[T]{}, fmt.Errorf("ivfq: row %d as encoded has squared norm %v", i, norm)
		}
		d.norms[i] = float32(norm)
	}
	return d, nil
}

// affine is the scale and offset mapping [lo, hi] onto codes [cmin, cmax].
func affine(lo, hi, cmin, cmax float64) (scale, offset float32) {
	scale = float32((hi - lo) / (cmax - cmin))
	return scale, float32(lo - float64(scale)*cmin)
}

// encodeCodes writes src as codes on scale and offset, clamped to
// [cmin, cmax].
func encodeCodes[C int8 | uint8](codes []C, src []float32, scale, offset float32, cmin, cmax float64) {
	for j, x := range src {
		c := cmin
		if scale != 0 {
			c = min(max(math.Round((float64(x)-float64(offset))/float64(scale)), cmin), cmax)
		}
		codes[j] = C(c)
	}
}

// row returns row i as float32: a view of Data for float32, otherwise
// decoded into dst, which it returns.
func (d Dense[T]) row(i int, dst []float32) []float32 {
	lo, hi := i*d.rows.Dims, (i+1)*d.rows.Dims
	switch codes := any(d.rows.Data).(type) {
	case []int8:
		return decodeCodes(dst, codes[lo:hi], d.scale, d.offset)
	case []uint8:
		return decodeCodes(dst, codes[lo:hi], d.scale, d.offset)
	}
	return d.rows.Row(i, dst)
}

// decoded returns the rows as float32, row-major: Data itself for float32,
// a new buffer otherwise.
func (d Dense[T]) decoded() []float32 {
	if data, ok := any(d.rows.Data).([]float32); ok {
		return data
	}
	out := make([]float32, len(d.rows.Data))
	for i := range d.norms {
		d.row(i, out[i*d.rows.Dims:])
	}
	return out
}

func decodeCodes[C int8 | uint8](dst []float32, codes []C, scale, offset float32) []float32 {
	dst = dst[:len(codes)]
	for i, c := range codes {
		dst[i] = float32(float64(offset) + float64(scale)*float64(c))
	}
	return dst
}

// Len is the number of rows.
func (d Dense[T]) Len() int { return len(d.norms) }

// Dims is the width of every row.
func (d Dense[T]) Dims() int { return d.rows.Dims }

// ScoreAll implements CentroidBlock. query must be finite with a finite
// float32 squared norm, as FloatTree's searches assume; otherwise scores
// are NaN. A finite query far from a row can still score -Inf, which ranks
// it last. It decodes a non-float32 block into scratch of its own on every
// call.
//
// ponytail: per-call decode makes non-float32 routing about 8-13x slower
// than float32 (measured at 4096x128 leaves, fanout 16, beam 16); pool the scratch or
// score codes without decoding if a profile asks.
func (d Dense[T]) ScoreAll(query, scores []float32) {
	if len(query) != d.rows.Dims || len(scores) < len(d.norms) {
		panic(fmt.Sprintf("ivfq: Dense.ScoreAll: %d-wide query and %d scores for %d rows of %d", len(query), len(scores), len(d.norms), d.rows.Dims))
	}
	scores = scores[:len(d.norms)]
	Dots(query, d.decoded(), scores)
	qq := float64(Dot(query, query))
	for i, dot := range scores {
		scores[i] = -float32(qq + float64(d.norms[i]) - 2*float64(dot))
	}
}
