package ivfq

import (
	"fmt"

	"github.com/axiomhq/ivfq/vector"
)

// DenseEmbedding is row i of a Dense[T], from Dense.At: a read-only view
// that keeps the block, with its scale and offset, alive. It is also the
// dense query every Score and ScoreAll in this module takes; to score a
// []float32 q, use NewDense[float32]([][]float32{q}) and At(0), once per
// query: At allocates its interface value on every call.
type DenseEmbedding[T vector.Scalar] struct {
	d Dense[T]
	i int
}

// Dims is the row's width.
func (e DenseEmbedding[T]) Dims() int { return e.d.rows.Dims }

// Score implements Embedding as Dense.ScoreAll scores this row.
func (e DenseEmbedding[T]) Score(query Embedding, metric Metric) (float32, error) {
	var s [1]float32
	err := e.d.score(query, metric, e.i, e.i+1, s[:])
	return s[0], err
}

// Decode copies the row's Dims decoded coordinates into dst, which must
// be at least that long, and leaves dst[Dims:] as it was. The values are
// what the block scores: float16 decoded as IEEE binary16, int8 and uint8
// as offset + scale*code. Decode keeps nothing of dst, and dst never
// aliases the block.
func (e DenseEmbedding[T]) Decode(dst []float32) error {
	dims := e.d.rows.Dims
	if len(dst) < dims {
		return fmt.Errorf("ivfq: Decode into %d values, want %d", len(dst), dims)
	}
	copy(dst, e.d.row(e.i, dst[:dims])) // a float32 row is a view of the block: copy it
	return nil
}
