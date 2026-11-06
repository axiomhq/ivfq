package ivfq

import (
	"context"
	"errors"
	"fmt"
)

// CentroidBlock is one tree node's centroid rows in an encoding the caller
// chose, held in the order its encoder received them. Len and Dims are its
// row count and width.
//
// ScoreAll sets scores[i], for every i < Len(), to the negated squared L2
// distance between query and row i, or an estimate of it: higher is
// nearer. A tree compares scores from different blocks, so every block of
// one tree must score that same quantity on the same scale. query is Dims()
// wide and scores has at least Len() entries; ScoreAll must not modify
// query, retain either slice, or write past scores[Len()-1]. A tree calls
// ScoreAll from concurrent searches, so a block must not change once built.
type CentroidBlock interface {
	Len() int
	Dims() int
	ScoreAll(query, scores []float32)
}

// Tree is FloatTree's hierarchy with every node's rows (child means for an
// internal node, leaf centroids for a leaf holder) held as one encoded
// block B: Tree[Dense[float32]], Tree[Dense[float16.Float16]],
// Tree[Dense[int8]], Tree[Dense[uint8]] or Tree[rabitq.Block]. The tree
// keeps the topology and, per row, the child node or original leaf id; the
// blocks keep everything else. There is no float32 copy beside the blocks,
// so a lossy B cannot return a leaf's original centroid; a caller that
// needs them keeps its own.
//
// A Tree is immutable and safe for concurrent searches.
type Tree[B CentroidBlock] struct {
	dims, leaves int
	nodes        []blockNode[B] // nodes[0] is the root
}

type blockNode[B CentroidBlock] struct {
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
func BuildTree[B CentroidBlock](ctx context.Context, centroids [][]float32, fanout int, encode func(rows [][]float32) (B, error)) (Tree[B], error) {
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
// keeps the beam best nodes per level by their blocks' scores and ranks
// every leaf of the holders it reaches, as FloatTree.EvaluationsBeam does;
// with B = Dense[float32] the result is EvaluationsBeam's. A q of the wrong
// width, or a beam or probe below one, returns nil. q must be finite.
func (t Tree[B]) Nearest(q []float32, beam, probe int) []int {
	if len(t.nodes) == 0 || beam <= 0 || probe <= 0 || len(q) != t.dims {
		return nil
	}
	var scores []float32
	score := func(b B) []float32 {
		if n := b.Len(); cap(scores) < n {
			scores = make([]float32, n)
		} else {
			scores = scores[:n]
		}
		b.ScoreAll(q, scores)
		return scores
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
			for j, s := range score(n.block) {
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
		for j, s := range score(n.block) {
			cands = append(cands, rankedNode{n.leaves[j], -s})
		}
	}
	return idsOf(selectNearest(nil, cands, probe))
}

// Dims is the width of every centroid in the tree.
func (t Tree[B]) Dims() int { return t.dims }

// Leaves is the number of leaf centroids the tree ranks over.
func (t Tree[B]) Leaves() int { return t.leaves }
