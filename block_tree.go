package ivfq

import (
	"context"
	"errors"
	"fmt"

	"github.com/axiomhq/ivfq/vector"
)

// Encoder encodes rows of S into a block B: DenseEncoder[S, T] makes
// Dense[T] blocks, rabitq.BlockEncoder[S] rabitq.Blocks, from any source
// scalar S. Encode borrows rows read-only for the call and returns a block
// that owns its data, rows in input order; rows must not change during the
// call.
//
// BuildTree does not take an Encoder: a tree's internal rows are fractional
// means held as [][]float32, so its encoder is NewDense[T] or a closure
// over rabitq.NewBlock.
type Encoder[S vector.Scalar, B EmbeddingBatch] interface {
	Encode(rows vector.Slice[S]) (B, error)
}

// Tree is FloatTree's hierarchy with every node's rows (child means for an
// internal node, leaf centroids for a leaf holder) held as one encoded
// block B: Tree[Dense[float32]], Tree[Dense[float16.Float16]],
// Tree[Dense[int8]], Tree[Dense[uint8]] or Tree[rabitq.Block]. A search
// scores each node's rows with B.ScoreAll under L2, whatever metric the
// field uses, and compares scores across blocks, so every block of one
// tree must score L2 on one scale. The tree keeps the topology and, per
// row, the child node or original leaf id; the blocks keep everything else.
// There is no float32 copy beside the blocks, so a lossy B cannot return a
// leaf's original centroid; a caller that needs them keeps its own.
//
// A Tree is immutable and safe for concurrent searches.
type Tree[B EmbeddingBatch] struct {
	dims, leaves int
	nodes        []blockNode[B] // nodes[0] is the root
}

type blockNode[B EmbeddingBatch] struct {
	block    B
	children []int // node id per row of an internal node
	leaves   []int // leaf id per row of a leaf holder
}

// BuildTree builds the hierarchy Build builds over centroids, from the same
// float32 working coordinates (each internal row is the mean of the leaves
// beneath it), then calls encode once per node with that node's rows and
// keeps only the blocks it returns. The rows are read-only and valid only
// during the call: a block copies what it keeps. Centroids follow Build's
// rules and must be at least one-dimensional. An encoder error, or a block
// whose Len or Dims disagrees with its rows, fails the build; so does a ctx
// cancelled at any point before BuildTree returns, even over no centroids.
func BuildTree[B EmbeddingBatch](ctx context.Context, centroids [][]float32, fanout int, encode func(rows [][]float32) (B, error)) (Tree[B], error) {
	if len(centroids) > 0 && len(centroids[0]) == 0 {
		return Tree[B]{}, errors.New("ivfq: centroids are zero-dimensional")
	}
	ft, err := buildTopology(ctx, centroids, fanout)
	if err != nil {
		return Tree[B]{}, err
	}
	t := Tree[B]{dims: ft.dims, leaves: len(centroids), nodes: make([]blockNode[B], len(ft.nodes))}
	for i, n := range ft.nodes {
		if err := ctx.Err(); err != nil {
			return Tree[B]{}, err
		}
		var rows [][]float32
		if len(n.children) > 0 {
			for _, c := range n.children {
				rows = append(rows, ft.nodes[c].centroid)
			}
		} else {
			for _, id := range n.leaves {
				rows = append(rows, centroids[id])
			}
		}
		b, err := encode(rows)
		if err != nil {
			return Tree[B]{}, fmt.Errorf("ivfq: encoding node %d: %w", i, err)
		}
		if b.Len() != len(rows) || b.Dims() != t.dims {
			return Tree[B]{}, fmt.Errorf("ivfq: node %d encoded %d rows of %d dims, want %d of %d", i, b.Len(), b.Dims(), len(rows), t.dims)
		}
		t.nodes[i] = blockNode[B]{block: b, children: n.children, leaves: n.leaves}
	}
	if err := ctx.Err(); err != nil { // cancelled during the last encode, or before an empty build
		return Tree[B]{}, err
	}
	return t, nil
}

// Nearest returns up to probe leaf ids, nearest first, ties by lower id. It
// keeps the beam best nodes per level by their blocks' L2 scores and ranks
// every leaf of the holders it reaches, as FloatTree.EvaluationsBeam does;
// with B = Dense[float32] and no score overflowing, the result is
// EvaluationsBeam's. An empty tree, or a beam or probe below one, returns
// nil, nil without looking at q. Otherwise the first error a block's
// ScoreAll returns (a q of the wrong width or representation, an infinite
// score) is Nearest's, with no ids.
func (t Tree[B]) Nearest(q Embedding, beam, probe int) ([]int, error) {
	if len(t.nodes) == 0 || beam <= 0 || probe <= 0 {
		return nil, nil
	}
	var scores []float32
	score := func(b B) ([]float32, error) {
		if n := b.Len(); cap(scores) < n {
			scores = make([]float32, n)
		} else {
			scores = scores[:n]
		}
		return scores, b.ScoreAll(q, L2, scores)
	}
	active, holders := []int{0}, []int(nil)
	var best, cands []rankedNode
	for len(active) > 0 {
		best = best[:0]
		for _, id := range active {
			n := &t.nodes[id]
			if len(n.children) == 0 {
				holders = append(holders, id)
				continue
			}
			s, err := score(n.block)
			if err != nil {
				return nil, err
			}
			for j, s := range s {
				best = insertNearest(best, rankedNode{n.children[j], -s}, beam)
			}
		}
		active = active[:0]
		for _, x := range best {
			active = append(active, x.id)
		}
	}
	for _, id := range holders {
		n := &t.nodes[id]
		s, err := score(n.block)
		if err != nil {
			return nil, err
		}
		for j, s := range s {
			cands = append(cands, rankedNode{n.leaves[j], -s})
		}
	}
	return idsOf(selectNearest(nil, cands, probe)), nil
}

// Dims is the width of every centroid in the tree.
func (t Tree[B]) Dims() int { return t.dims }

// Leaves is the number of leaf centroids the tree ranks over.
func (t Tree[B]) Leaves() int { return t.leaves }
