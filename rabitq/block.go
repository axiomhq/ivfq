package rabitq

import "github.com/axiomhq/ivfq"

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

// NewBlock encodes rows for squared-L2 routing in rot's frame. With rot
// fixed it is BuildTree's encoder:
//
//	func(rows [][]float32) (rabitq.Block, error) { return rabitq.NewBlock(rows, rot) }
func NewBlock(rows [][]float32, rot *Rotation) (Block, error) {
	// Not PackFastScan: at fanout 16 it padded a tree's codes to 1.6x
	// their bytes for about 10% faster routing.
	col, err := Quantize(rows, Options{Metric: ivfq.L2, Rotation: rot})
	return Block{col: col, rot: rot}, err
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
