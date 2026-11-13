package ivfq

import (
	"errors"
	"fmt"
	"math"

	"github.com/axiomhq/ivfq/vector"
	"github.com/x448/float16"
)

// Dense is an EmbeddingBatch of rows stored as T and decoded to float32 to
// score. NewDense[T] is its encoder for BuildTree.
//
// float32 rows are kept exactly. float16 rows are rounded to the nearest
// IEEE binary16 value; a coordinate that rounds past ±65504 (|x| ≥ 65520)
// is an error, not an infinity. int8 and uint8 rows are codes on one
// affine scale per block, because tree means are fractional: the block's
// [min, max] coordinate range maps onto all 256 codes, scale =
// (max-min)/255 and offset = min - scale*(T's minimum), a coordinate x is
// stored as round((x-offset)/scale), halves away from zero, clamped to
// T's range, and decodes to offset + scale*code, within scale/2 of x up
// to float32 rounding. A block whose coordinates are all equal decodes
// them exactly.
//
// Scores are of the decoded rows, not the coordinates before encoding,
// exactly as a Dense[float32] of the decoded rows would score them:
// -(‖q‖² + ‖row‖² - 2·q·row) for L2, from float32 q·row and squared norms
// combined in float64 (FloatTree's ranking distance negated); q·row for
// InnerProduct; and q·row/(‖q‖‖row‖) from the same terms for Cosine,
// except that a row or query whose squared norm is under 2⁻⁶⁰ is summed
// wholly in float64, so a tiny nonzero vector is not mistaken for zero.
// Cosine is 0 when either vector is exactly zero.
type Dense[T vector.Scalar] struct {
	rows          vector.Slice[T]
	norms         []float32 // squared norm of each decoded row
	scale, offset float32   // integer T: decoded = offset + scale*code
	lut           []float32 // integer T: decoded value of each code byte
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
		d.lut = codeTable(func(c int) float64 { return float64(int8(c)) }, d.scale, d.offset)
	case []uint8:
		d.scale, d.offset = affine(lo, hi, 0, 255)
		d.lut = codeTable(func(c int) float64 { return float64(uint8(c)) }, d.scale, d.offset)
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
		return decodeCodes(dst, codes[lo:hi], d.lut)
	case []uint8:
		return decodeCodes(dst, codes[lo:hi], d.lut)
	}
	return d.rows.Row(i, dst)
}

// codeTable is every code byte's decoded value, offset + scale*code,
// computed in float64 and rounded once.
func codeTable(code func(c int) float64, scale, offset float32) []float32 {
	t := make([]float32, 256)
	for c := range t {
		t[c] = float32(float64(offset) + float64(scale)*code(c))
	}
	return t
}

func decodeCodes[C int8 | uint8](dst []float32, codes []C, lut []float32) []float32 {
	dst = dst[:len(codes)]
	lut = lut[:256]
	for i, c := range codes {
		dst[i] = lut[uint8(c)]
	}
	return dst
}

// Len is the number of rows.
func (d Dense[T]) Len() int { return len(d.norms) }

// Dims is the width of every row.
func (d Dense[T]) Dims() int { return d.rows.Dims }

// At implements EmbeddingBatch.
func (d Dense[T]) At(i int) Embedding {
	if i < 0 || i >= len(d.norms) {
		panic(fmt.Sprintf("ivfq: Dense.At(%d) of %d rows", i, len(d.norms)))
	}
	return DenseEmbedding[T]{d: d, i: i}
}

// ScoreAll implements EmbeddingBatch for a dense query, a DenseEmbedding
// of any scalar, under L2, Cosine or InnerProduct. It reads a float32
// query in place and decodes any other, and a non-float32 block a row at a
// time, into scratch of its own. A finite query far from a row can
// overflow to an infinite score, which is an error.
func (d Dense[T]) ScoreAll(query Embedding, metric Metric, dst []float32) error {
	return d.score(query, metric, 0, len(d.norms), dst)
}

// score sets dst[i-lo] to the score of row i, for lo <= i < hi.
func (d Dense[T]) score(query Embedding, metric Metric, lo, hi int, dst []float32) error {
	dims := d.rows.Dims
	if metric != L2 && metric != Cosine && metric != InnerProduct {
		return fmt.Errorf("ivfq: Dense cannot score %v", metric)
	}
	if dims == 0 {
		return errors.New("ivfq: zero Dense")
	}
	if len(dst) < hi-lo {
		return fmt.Errorf("ivfq: %d scores for %d rows", len(dst), hi-lo)
	}
	q, err := denseQuery(query, dims)
	if err != nil {
		return err
	}
	dst = dst[:hi-lo]
	if data, ok := any(d.rows.Data).([]float32); ok {
		Dots(q, data[lo*dims:hi*dims], dst)
	} else {
		row := make([]float32, dims)
		for i := range dst {
			dst[i] = Dot(q, d.row(lo+i, row))
		}
	}
	qq := float64(Dot(q, q))
	switch metric {
	case L2:
		for i, dot := range dst {
			dst[i] = -float32(qq + float64(d.norms[lo+i]) - 2*float64(dot))
		}
	case Cosine:
		var row []float32
		for i, dot := range dst {
			rr := float64(d.norms[lo+i])
			if qq >= tinyNorm2 && rr >= tinyNorm2 {
				dst[i] = float32(float64(dot) / math.Sqrt(qq*rr))
				continue
			}
			// Under tinyNorm2: sum this row in float64, so that only a
			// true zero scores 0.
			if row == nil {
				row = make([]float32, dims)
			}
			dst[i] = cosine64(q, d.row(lo+i, row))
		}
	}
	for i, s := range dst {
		if math.IsNaN(float64(s)) || math.IsInf(float64(s), 0) {
			return fmt.Errorf("ivfq: row %d scores %v", lo+i, s)
		}
	}
	return nil
}

// tinyNorm2 is the squared norm under which cosine is summed in float64.
// Above it, for both vectors, the float32 dot product's underflow error
// (at most dims·2⁻¹⁴⁹) is negligible against ‖q‖‖row‖ ≥ 2⁻⁶⁰.
const tinyNorm2 = 0x1p-60

// cosine64 is the cosine of q and r summed in float64, 0 when either is
// exactly zero.
func cosine64(q, r []float32) float32 {
	var dot, qq, rr float64
	for j, x := range r {
		dot += float64(q[j]) * float64(x)
		qq += float64(q[j]) * float64(q[j])
		rr += float64(x) * float64(x)
	}
	if qq == 0 || rr == 0 {
		return 0
	}
	return float32(dot / (math.Sqrt(qq) * math.Sqrt(rr)))
}

// denseQuery returns query, which must be a dims-wide DenseEmbedding, as
// float32 values: a read-only view for float32, decoded into a new buffer
// otherwise.
func denseQuery(query Embedding, dims int) ([]float32, error) {
	// The type switch comes first: an unsupported query, a typed nil
	// pointer included, is rejected before any of its methods runs. ivfq
	// and rabitq each keep a switch like this one (a new vector.Scalar
	// needs both); rabitq reads queries only through DenseEmbedding.Decode,
	// the one cross-package seam. There is deliberately no registry.
	switch e := query.(type) {
	case DenseEmbedding[float32]:
		if e.Dims() != dims {
			return decodeQuery(e, dims) // its width error
		}
		// Read in place for the call, not copied: a tree scores one query
		// against every node it visits.
		return e.d.row(e.i, nil), nil
	case DenseEmbedding[float16.Float16]:
		return decodeQuery(e, dims)
	case DenseEmbedding[int8]:
		return decodeQuery(e, dims)
	case DenseEmbedding[uint8]:
		return decodeQuery(e, dims)
	}
	return nil, fmt.Errorf("ivfq: cannot score a %T query", query)
}

func decodeQuery[T vector.Scalar](e DenseEmbedding[T], dims int) ([]float32, error) {
	if e.Dims() != dims {
		return nil, fmt.Errorf("ivfq: %d-wide query for %d-wide rows", e.Dims(), dims)
	}
	q := make([]float32, dims)
	return q, e.Decode(q)
}
