package rabitq

import (
	"errors"
	"fmt"
	"math"

	"github.com/axiomhq/ivfq"
	"github.com/axiomhq/ivfq/vector"
	"github.com/x448/float16"
)

// Block is an ivfq.EmbeddingBatch of RaBitQ codes: Dims bits and two
// float32 per row, plus the block's own mean as the codes' encoding
// centroid, and no float32 copy of the rows. The codes are L2 codes: a
// Block scores ivfq.L2 only. ScoreAll is the RaBitQ estimate of
// -‖q - row‖² (not its bound, and not exact even over every row), the
// score ivfq.Tree routes by, so routing over Blocks is approximate: a
// branch dropped on an estimate is not recovered by exactly reranking the
// leaves the search did reach. (Query.Exact drops the bound, which
// ScoreAll never reads; it does not remove the estimate's error.)
type Block struct {
	col Quantizer
	rot *Rotation
}

// NewBlock encodes rows, which must be finite, for squared-L2 routing in
// rot's frame. With rot fixed it is BuildTree's encoder:
//
//	func(rows [][]float32) (rabitq.Block, error) { return rabitq.NewBlock(rows, rot) }
func NewBlock(rows [][]float32, rot *Rotation) (Block, error) {
	for i, r := range rows {
		if err := checkFinite(i, r); err != nil {
			return Block{}, err
		}
	}
	// Not PackFastScan: at fanout 16 it padded a tree's codes to 1.6x
	// their bytes for about 10% faster routing.
	col, err := Quantize(rows, Options{Metric: ivfq.L2, Rotation: rot})
	return Block{col: col, rot: rot}, err
}

// BlockEncoder is the ivfq.Encoder from rows of S to Blocks in Rotation's
// frame. Encode reads S's values as float32 (float16 decoded as IEEE
// binary16), which must be finite, and encodes them as NewBlock does, bit
// for bit, a row at a time: it borrows rows only for the call, keeps
// nothing of them, and never builds a float32 copy of the whole input.
// Like NewBlock it accepts zero rows.
type BlockEncoder[S vector.Scalar] struct {
	Rotation *Rotation
}

// Encode implements ivfq.Encoder.
func (e BlockEncoder[S]) Encode(rows vector.Slice[S]) (Block, error) {
	n, err := rows.Rows()
	if err != nil {
		return Block{}, err
	}
	if err := checkFrame(Options{Metric: ivfq.L2, Rotation: e.Rotation}); err != nil {
		return Block{}, err
	}
	if rows.Dims != e.Rotation.dims {
		return Block{}, fmt.Errorf("rabitq: rows are %d-wide, the rotation is %d", rows.Dims, e.Rotation.dims)
	}
	scratch := make([]float32, rows.Dims)
	for i := range n {
		if err := checkFinite(i, rows.Row(i, scratch)); err != nil {
			return Block{}, err
		}
	}
	code := quantize(n, rows.Dims, func(i int) []float32 { return rows.Row(i, scratch) }, false, e.Rotation, 0)
	return Block{col: Quantizer{Dims: rows.Dims, Code: code}, rot: e.Rotation}, nil
}

// Len is the number of rows, 0 for the zero Block.
func (b Block) Len() int {
	if b.col.Code == nil {
		return 0
	}
	return b.col.Rows()
}

// Dims is the width of every row.
func (b Block) Dims() int { return b.col.Dims }

// At implements ivfq.EmbeddingBatch.
func (b Block) At(i int) ivfq.Embedding {
	if i < 0 || i >= b.Len() {
		panic(fmt.Sprintf("rabitq: Block.At(%d) of %d rows", i, b.Len()))
	}
	return BlockEmbedding{b: b, i: i}
}

// ScoreAll implements ivfq.EmbeddingBatch for a dense query (an
// ivfq.DenseEmbedding of any scalar) under ivfq.L2. Each call decodes the
// query and prepares it against this block's centroid and rotation, a
// Scorer of its own, so concurrent calls share nothing mutable: O(Dims log
// Dims) per block, not once per search.
func (b Block) ScoreAll(query ivfq.Embedding, metric ivfq.Metric, dst []float32) error {
	return b.score(query, metric, 0, b.Len(), dst)
}

// score sets dst[i-lo] to row i's estimate, for lo <= i < hi.
func (b Block) score(query ivfq.Embedding, metric ivfq.Metric, lo, hi int, dst []float32) error {
	if metric != ivfq.L2 {
		return fmt.Errorf("rabitq: a Block holds L2 codes and cannot score %v", metric)
	}
	if b.rot == nil {
		return errors.New("rabitq: zero Block")
	}
	if len(dst) < hi-lo {
		return fmt.Errorf("rabitq: %d scores for %d rows", len(dst), hi-lo)
	}
	q, err := denseQuery(query, b.Dims())
	if err != nil {
		return err
	}
	s := b.col.Scorer(NewQuery(q, ivfq.L2, b.rot))
	for i := range dst[:hi-lo] {
		x := s.Score(lo + i)
		if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
			return fmt.Errorf("rabitq: row %d scores %v", lo+i, x)
		}
		dst[i] = x
	}
	return nil
}

// denseQuery decodes query, which must be a dims-wide ivfq.DenseEmbedding,
// into a new buffer.
func denseQuery(query ivfq.Embedding, dims int) ([]float32, error) {
	// The type switch comes first: an unsupported query, a typed nil
	// pointer included, is rejected before any of its methods runs. ivfq
	// and rabitq each keep a switch like this one (a new vector.Scalar
	// needs both); rabitq reads queries only through DenseEmbedding.Decode,
	// the one cross-package seam. There is deliberately no registry.
	switch e := query.(type) {
	case ivfq.DenseEmbedding[float32]:
		return decodeQuery(e, dims)
	case ivfq.DenseEmbedding[float16.Float16]:
		return decodeQuery(e, dims)
	case ivfq.DenseEmbedding[int8]:
		return decodeQuery(e, dims)
	case ivfq.DenseEmbedding[uint8]:
		return decodeQuery(e, dims)
	}
	return nil, fmt.Errorf("rabitq: cannot score a %T query", query)
}

func decodeQuery[T vector.Scalar](e ivfq.DenseEmbedding[T], dims int) ([]float32, error) {
	if e.Dims() != dims {
		return nil, fmt.Errorf("rabitq: %d-wide query for %d-wide rows", e.Dims(), dims)
	}
	q := make([]float32, dims)
	return q, e.Decode(q)
}

// checkFinite reports row i's first NaN or infinite coordinate.
func checkFinite(i int, row []float32) error {
	for j, x := range row {
		if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
			return fmt.Errorf("rabitq: row %d coordinate %d is %v", i, j, x)
		}
	}
	return nil
}
