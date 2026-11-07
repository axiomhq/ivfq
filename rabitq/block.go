package rabitq

import (
	"fmt"
	"math"

	"github.com/axiomhq/ivfq"
	"github.com/axiomhq/ivfq/vector"
)

// Block is an ivfq.CentroidBlock of RaBitQ codes: Dims bits and two
// float32 per row, plus the block's own mean as the codes' encoding
// centroid, and no float32 copy of the rows. ScoreAll is the RaBitQ
// estimate of -‖q - row‖², the score ivfq.Tree routes by, so routing over
// Blocks is approximate: a branch dropped on an estimate is not recovered
// by exactly reranking the leaves the search did reach. (Query.Exact drops
// the bound, which ScoreAll never reads; it does not remove the estimate's
// error.)
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
	code := quantize(n, rows.Dims, func(i int) []float32 { return rows.Row(i, scratch) }, false, e.Rotation)
	return Block{col: Quantizer{Dims: rows.Dims, Code: code}, rot: e.Rotation}, nil
}

// Len is the number of rows.
func (b Block) Len() int { return b.col.Rows() }

// Dims is the width of every row.
func (b Block) Dims() int { return b.col.Dims }

// ScoreAll implements ivfq.CentroidBlock with a Scorer of its own per
// call, so concurrent searches share nothing mutable. Each call prepares
// the query against this block's centroid and rotation: O(Dims log Dims)
// per block, not once per search.
func (b Block) ScoreAll(query, scores []float32) {
	s := b.col.Scorer(NewQuery(query, ivfq.L2, b.rot))
	s.ScoreAll(scores, nil)
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
