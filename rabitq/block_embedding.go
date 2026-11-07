package rabitq

import "github.com/axiomhq/ivfq"

// BlockEmbedding is row i of a Block, from Block.At: a read-only view
// that keeps the block's codes and frame alive. It holds codes only, no
// coordinates, so nothing scores it as a query.
type BlockEmbedding struct {
	b Block
	i int
}

// Dims is the row's width.
func (e BlockEmbedding) Dims() int { return e.b.Dims() }

// Score implements ivfq.Embedding as Block.ScoreAll scores this row.
func (e BlockEmbedding) Score(query ivfq.Embedding, metric ivfq.Metric) (float32, error) {
	var s [1]float32
	err := e.b.score(query, metric, e.i, e.i+1, s[:])
	return s[0], err
}
