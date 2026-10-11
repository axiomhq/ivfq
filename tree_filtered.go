package ivfq

import (
	"slices"
	"sync"
)

// FilteredNearest is EvaluationsBeamI8Work restricted to a subset of the
// leaves: up to probe of the eligible leaf ids, nearest first, and the
// centroid work it took. Leaves outside eligible are never scored, and
// neither is any internal node with no eligible leaf beneath it.
//
// The search is the beam's, over eligible subtrees only. Each level keeps
// at least beam of its nearest eligible nodes, and past beam keeps taking
// the next nearest until the kept nodes and the holders already reached
// hold want eligible leaves:
//
//	want = max(2*probe, beam*fanout)
//
// beam*fanout is what the unfiltered beam reaches at the leaf level (beam
// holders of up to fanout leaves); the filtered search reaches as many
// eligible leaves, however sparse the filter is under the nearest nodes, so
// a rare or clustered filter never ranks probe leaves out of a short pool.
// When want covers every eligible leaf the internal levels are skipped and
// every eligible leaf ranked, which is the same pool with no internal
// work. The leaves reached are ranked exactly (in float32, or through the
// holders' int8 copies with EvaluationsBeamI8's bound when every reached
// holder has one), so only which subtrees the beam dropped can cost recall.
//
// Building the eligibility mask is one pass over the tree's node and leaf
// lists, integer work and no distance. Ids outside [0, Leaves()) and
// leaves no holder owns (Upsert/Truncate can leave some) are never
// returned; duplicates count once.
func (t Tree) FilteredNearest(q []float32, eligible []int, beam, probe int) ([]int, TreeWork) {
	if t.root < 0 || probe <= 0 || beam <= 0 || len(q) != t.dims || len(eligible) == 0 {
		return nil, TreeWork{}
	}
	w := filterWorkspaces.Get().(*filterWorkspace)
	defer filterWorkspaces.Put(w)
	total := w.mask(t, eligible)
	if total == 0 {
		return nil, TreeWork{}
	}
	want := max(2*probe, beam*t.fanout)
	var work TreeWork
	tw := &w.treeWorkspace
	if want >= total {
		tw.holders = tw.holders[:0]
		for id := range t.nodes {
			if len(t.nodes[id].children) == 0 && w.cnt[id] > 0 {
				tw.holders = append(tw.holders, id)
			}
		}
	} else {
		work.Float32 = t.filteredHolders(w, q, beam, want)
	}
	var out []rankedNode
	var lw TreeWork
	if t.holdersQuantized(tw.holders) {
		out, lw = t.rankHoldersI8(tw, q, probe, w.keep)
	} else {
		out, lw = t.rankHoldersF32(w, q, probe)
	}
	work.Float32 += lw.Float32
	work.Quantized += lw.Quantized
	if len(out) == 0 {
		return nil, work
	}
	ids := make([]int, len(out))
	for i, x := range out {
		ids[i] = x.id
	}
	return ids, work
}

// filterWorkspace is a filtered search's scratch: the tree workspace the
// leaf rankers share, the eligibility of every leaf (keep) and the number
// of eligible leaves under every node (cnt).
type filterWorkspace struct {
	treeWorkspace
	keep []bool
	cnt  []int32
}

var filterWorkspaces = sync.Pool{New: func() any { return new(filterWorkspace) }}

// mask sets keep from eligible and cnt for every node, and returns the
// eligible leaves the tree's holders own. Children are numbered after their
// parents (Build appends a node before its children, UnmarshalTree rejects
// any other order, and Upsert adds no nodes to a non-empty tree), so one
// pass from the last node up has every child's count before its parent's.
func (w *filterWorkspace) mask(t Tree, eligible []int) int {
	if cap(w.keep) < len(t.leaves) {
		w.keep = make([]bool, len(t.leaves))
	}
	w.keep = w.keep[:len(t.leaves)]
	clear(w.keep)
	for _, id := range eligible {
		if id >= 0 && id < len(w.keep) {
			w.keep[id] = true
		}
	}
	if cap(w.cnt) < len(t.nodes) {
		w.cnt = make([]int32, len(t.nodes))
	}
	w.cnt = w.cnt[:len(t.nodes)]
	for i := len(t.nodes) - 1; i >= 0; i-- {
		n := &t.nodes[i]
		var c int32
		for _, leaf := range n.leaves {
			if w.keep[leaf] {
				c++
			}
		}
		for _, child := range n.children {
			c += w.cnt[child]
		}
		w.cnt[i] = c
	}
	return int(w.cnt[t.root])
}

// filteredHolders is the beam over eligible internal nodes: w.holders is
// the eligible holders it reaches, and it returns the float32 distances it
// computed. A level keeps its nearest beam nodes, then more, nearest first,
// until the holders reached and the nodes kept hold want eligible leaves.
func (t Tree) filteredHolders(w *filterWorkspace, q []float32, beam, want int) int {
	tw := &w.treeWorkspace
	tw.active = append(tw.active[:0], t.root)
	tw.holders = tw.holders[:0]
	evals, reached := 0, 0
	qq := Dot(q, q)
	for len(tw.active) > 0 {
		tw.next = tw.next[:0]
		for _, id := range tw.active {
			n := &t.nodes[id]
			if len(n.children) == 0 {
				tw.holders = append(tw.holders, id)
				reached += int(w.cnt[id])
				continue
			}
			whole := true
			for _, c := range n.children {
				if w.cnt[c] == 0 {
					whole = false
					break
				}
			}
			if whole {
				dots := tw.dotsOf(q, n.block, len(n.children))
				for j, c := range n.children {
					d := float64(qq) + float64(n.norms[j]) - 2*float64(dots[j])
					tw.next = append(tw.next, rankedNode{c, float32(d)})
				}
				evals += len(n.children)
				continue
			}
			for j, c := range n.children {
				if w.cnt[c] == 0 {
					continue
				}
				d := float64(qq) + float64(n.norms[j]) - 2*float64(Dot(q, n.block[j*t.dims:(j+1)*t.dims]))
				tw.next = append(tw.next, rankedNode{c, float32(d)})
				evals++
			}
		}
		if len(tw.next) == 0 {
			break
		}
		slices.SortFunc(tw.next, func(a, b rankedNode) int {
			switch {
			case a.before(b):
				return -1
			case b.before(a):
				return 1
			}
			return 0
		})
		tw.active = tw.active[:0]
		acc := reached
		for i, x := range tw.next {
			if i >= beam && acc >= want {
				break
			}
			tw.active = append(tw.active, x.id)
			acc += int(w.cnt[x.id])
		}
	}
	return evals
}

// rankHoldersF32 ranks the eligible leaves of w.holders in float32: the
// count nearest, nearest first, ties by id.
func (t Tree) rankHoldersF32(w *filterWorkspace, q []float32, count int) ([]rankedNode, TreeWork) {
	var work TreeWork
	tw := &w.treeWorkspace
	qq := Dot(q, q)
	tw.next = tw.next[:0]
	for _, id := range tw.holders {
		n := &t.nodes[id]
		if int(w.cnt[id]) == len(n.leaves) {
			dots := tw.dotsOf(q, n.block, len(n.leaves))
			for j, leaf := range n.leaves {
				d := float64(qq) + float64(n.norms[j]) - 2*float64(dots[j])
				tw.next = append(tw.next, rankedNode{leaf, float32(d)})
			}
			work.Float32 += len(n.leaves)
			continue
		}
		for j, leaf := range n.leaves {
			if !w.keep[leaf] {
				continue
			}
			d := float64(qq) + float64(n.norms[j]) - 2*float64(Dot(q, n.block[j*t.dims:(j+1)*t.dims]))
			tw.next = append(tw.next, rankedNode{leaf, float32(d)})
			work.Float32++
		}
	}
	tw.best = selectNearest(tw.best[:0], tw.next, count)
	return tw.best, work
}
