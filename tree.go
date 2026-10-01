package ivfq

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"github.com/axiomhq/ivfq/kmeans"
	"math"
	"slices"
)

type treeNode struct {
	centroid []float32
	children []int
	leaves   []int
	// sub holds the id of every leaf holder beneath an internal node, in
	// child order. Derived by index alongside the block, never serialized;
	// it is what the two-stage assigner's second stage walks to scan a
	// whole subtree's leaves without a pointer chase.
	sub []int
	// block holds the centroids this node ranks over, contiguous and in
	// children order (internal node) or leaves order (leaf holder), with
	// their squared norms alongside. Derived by index after Build or
	// UnmarshalTree, never serialized: a beam step is then one sequential
	// scan and one dot product per candidate instead of a pointer chase
	// and a sqrt per candidate.
	block []float32
	norms []float32
}

// Tree is a deterministic hierarchy over a set of leaf centroids.
type Tree struct {
	dims, fanout, root, height int
	nodes                      []treeNode
	leaves                     [][]float32
}

// Build constructs a fanout-ary hierarchy. Its fixed seeds make identical
// inputs deterministic; fanout values below two are promoted to two. A
// cancelled ctx returns its error and no tree.
func Build(ctx context.Context, centroids [][]float32, fanout int) (Tree, error) {
	t := Tree{fanout: max(2, fanout), root: -1, leaves: centroids}
	if len(centroids) == 0 {
		return t, nil
	}
	t.dims = len(centroids[0])
	ids := make([]int, len(centroids))
	for i := range ids {
		ids[i] = i
	}
	var err error
	if t.root, t.height, err = t.build(ctx, ids, 0); err != nil {
		return Tree{}, err
	}
	t.index()
	return t, nil
}

// index derives every node's contiguous scan block. One arena holds every
// leaf centroid in holder order, so a holder's block is a sub-slice of it
// and t.leaves keeps addressing the same floats by id: the tree's memory
// stays one copy of the leaves plus the internal centroids, close to its
// serialized size, which is what the object cache charges for it.
func (t *Tree) index() {
	if t.dims == 0 || len(t.nodes) == 0 {
		return
	}
	arena := make([]float32, 0, len(t.leaves)*t.dims)
	leaves := make([][]float32, len(t.leaves))
	for i := range t.nodes {
		n := &t.nodes[i]
		if len(n.children) > 0 {
			n.block = make([]float32, 0, len(n.children)*t.dims)
			n.norms = make([]float32, len(n.children))
			for j, id := range n.children {
				c := t.nodes[id].centroid
				n.block = append(n.block, c...)
				n.norms[j] = Dot(c, c)
			}
			continue
		}
		start := len(arena)
		n.norms = make([]float32, len(n.leaves))
		for j, id := range n.leaves {
			v := t.leaves[id]
			arena = append(arena, v...) // never reallocates: the arena is sized for every leaf
			n.norms[j] = Dot(v, v)
		}
		n.block = arena[start:len(arena):len(arena)]
		for j, id := range n.leaves {
			leaves[id] = n.block[j*t.dims : (j+1)*t.dims : (j+1)*t.dims]
		}
	}
	for id, v := range leaves {
		if v == nil {
			leaves[id] = t.leaves[id] // unowned by any holder; unreachable from search
		}
	}
	t.leaves = leaves
	// Descendant holder lists, children numbered after parents so one
	// descending pass has every child's list ready. Holders leave sub nil:
	// their own leaves are the stage-two scan.
	subs := make([][]int, len(t.nodes))
	for i := len(t.nodes) - 1; i >= 0; i-- {
		n := &t.nodes[i]
		if len(n.children) == 0 {
			continue
		}
		total := 0
		for _, c := range n.children {
			if len(t.nodes[c].children) == 0 {
				total++
			} else {
				total += len(subs[c])
			}
		}
		s := make([]int, 0, total)
		for _, c := range n.children {
			if len(t.nodes[c].children) == 0 {
				s = append(s, c)
			} else {
				s = append(s, subs[c]...)
			}
		}
		subs[i] = s
		n.sub = s
	}
}

func (t *Tree) build(ctx context.Context, ids []int, depth int) (int, int, error) {
	if err := ctx.Err(); err != nil {
		return 0, 0, err
	}
	n := treeNode{centroid: meanIDs(t.leaves, ids)}
	idx := len(t.nodes)
	t.nodes = append(t.nodes, n)
	if len(ids) <= t.fanout {
		t.nodes[idx].leaves = append([]int(nil), ids...)
		return idx, 1, nil
	}
	vs := make([][]float32, len(ids))
	for i, id := range ids {
		vs[i] = t.leaves[id]
	}
	k := min(t.fanout, (len(ids)+t.fanout-1)/t.fanout)
	_, assign, err := (kmeans.Config{K: k, Iters: 6, Seed: int64(0x51f15e + depth)}).Fit(ctx, vs)
	if err != nil {
		return 0, 0, err
	}
	groups := make([][]int, k)
	for i, g := range assign {
		groups[g] = append(groups[g], ids[i])
	}
	// Identical centroids all tie to assignment zero. Split them by stable
	// input order so degenerate data still makes progress.
	if len(groups[0]) == len(ids) {
		for i := range groups {
			lo, hi := i*len(ids)/k, (i+1)*len(ids)/k
			groups[i] = ids[lo:hi]
		}
	}
	h := 0
	for _, g := range groups {
		if len(g) == 0 {
			continue
		}
		child, ch, err := t.build(ctx, g, depth+1)
		if err != nil {
			return 0, 0, err
		}
		t.nodes[idx].children = append(t.nodes[idx].children, child)
		h = max(h, ch)
	}
	return idx, h + 1, nil
}

func meanIDs(v [][]float32, ids []int) []float32 {
	m := make([]float32, len(v[ids[0]]))
	for _, id := range ids {
		for j, x := range v[id] {
			m[j] += x
		}
	}
	for j := range m {
		m[j] /= float32(len(ids))
	}
	return m
}

// Nearest returns up to probe leaf centroid indices, nearest first.
func (t Tree) Nearest(q []float32, probe int) []int {
	ids, _ := t.nearest(q, probe)
	return ids
}

// Evaluations is Nearest with the number of centroid distance evaluations.
func (t Tree) Evaluations(q []float32, probe int) ([]int, int) { return t.nearest(q, probe) }

// Searcher is Nearest, Assign and TwoStageAssign with scratch that
// survives between calls: a warm Searcher allocates nothing per query. It
// belongs to one goroutine; a bulk caller makes one per worker. Nearest
// allocates its candidate buffer per call, about 45 KB per row when a
// re-route runs every member of a cluster through it.
type Searcher struct {
	t   *Tree
	w   treeWorkspace
	out []int
}

// NewSearcher returns a Searcher over t. t must not change while the
// Searcher is in use.
func (t *Tree) NewSearcher() *Searcher { return &Searcher{t: t} }

// Nearest is Tree.Nearest: up to probe leaf ids, nearest first, in the
// same order. The slice is the Searcher's own and the next call overwrites
// it.
func (s *Searcher) Nearest(q []float32, probe int) []int {
	best, _ := s.t.nearestInto(&s.w, q, probe, probe)
	s.out = s.out[:0]
	for _, x := range best {
		s.out = append(s.out, x.id)
	}
	return s.out
}

// Assign is the leaf a beam-wide search ranks first, or -1 when the tree
// is empty or q is the wrong width: the same leaf Nearest(q, beam)[0]
// names.
func (s *Searcher) Assign(q []float32, beam int) int {
	best, _ := s.t.nearestInto(&s.w, q, beam, 1)
	if len(best) == 0 {
		return -1
	}
	return best[0].id
}

// TwoStageAssign ranks one leaf in two exact stages: a distance for every
// child of the root, then a distance for every leaf under the top
// subtrees. It is TwoStageNearest(q, top)[0], or -1.
//
// Measured, it is a wash for bulk routing: on real BIGANN rows the stage-two
// scan is the whole cost and the beam's pruning wins on large trees, so the
// maintainer's router keeps the beam. Small trees are where this wins: at
// k = 977 a top-4 two-stage reached 99.1% agreement at 2.5x less wall per
// row than beam 16, because the stage-one fan there is 10 wide and four
// subtrees cover nearly the whole set. Choose by the benchmark, not by the
// evaluation count.
func (s *Searcher) TwoStageAssign(q []float32, top int) int {
	best, _ := s.t.twoStageInto(&s.w, q, top, 1)
	if len(best) == 0 {
		return -1
	}
	return best[0].id
}

type treeWorkspace struct {
	active, holders []int
	next, best      []rankedNode
	dots            []float32
}

// dots returns q's dot product with each of the first n rows of block,
// dims wide (Dots: Dot per row, bit for bit, one call per
// node instead of one per centroid), in scratch the next call reuses.
func (w *treeWorkspace) dotsOf(q, block []float32, n int) []float32 {
	if cap(w.dots) < n {
		w.dots = make([]float32, n)
	}
	w.dots = w.dots[:n]
	Dots(q, block[:n*len(q)], w.dots)
	return w.dots
}

func (t Tree) nearest(q []float32, probe int) ([]int, int) {
	if t.root < 0 || probe <= 0 || len(q) != t.dims {
		return nil, 0
	}
	w := treeWorkspace{next: make([]rankedNode, 0, min(t.fanout, len(t.nodes))), best: make([]rankedNode, 0, probe)}
	best, evals := t.nearestInto(&w, q, probe, probe)
	out := make([]int, len(best))
	for i, x := range best {
		out[i] = x.id
	}
	return out, evals
}

func (t Tree) nearestInto(w *treeWorkspace, q []float32, beam, count int) ([]rankedNode, int) {
	if t.root < 0 || beam <= 0 || count <= 0 || len(q) != t.dims {
		return nil, 0
	}
	w.active = append(w.active[:0], t.root)
	w.holders = w.holders[:0]
	evals := 0
	qq := Dot(q, q)
	// Squared distances use the norm identity, sharing q's norm across
	// each block. The terms combine in float64 so near ties retain their
	// low bits on long vectors.
	scan := func(n *treeNode, ids []int) {
		dots := w.dotsOf(q, n.block, len(ids))
		for j, id := range ids {
			d := float64(qq) + float64(n.norms[j]) - 2*float64(dots[j])
			w.next = append(w.next, rankedNode{id, float32(d)})
		}
		evals += len(ids)
	}
	for len(w.active) > 0 {
		w.best = w.best[:0]
		for _, id := range w.active {
			n := &t.nodes[id]
			if len(n.children) == 0 {
				w.holders = append(w.holders, id)
				continue
			}
			dots := w.dotsOf(q, n.block, len(n.children))
			for j, child := range n.children {
				d := float64(qq) + float64(n.norms[j]) - 2*float64(dots[j])
				w.best = insertNearest(w.best, rankedNode{child, float32(d)}, beam)
			}
			evals += len(n.children)
		}
		if len(w.best) == 0 {
			break
		}
		w.active = w.active[:0]
		for _, x := range w.best {
			w.active = append(w.active, x.id)
		}
	}
	if count == 1 {
		var best rankedNode
		have := false
		for _, id := range w.holders {
			n := &t.nodes[id]
			dots := w.dotsOf(q, n.block, len(n.leaves))
			for j, leaf := range n.leaves {
				d := float64(qq) + float64(n.norms[j]) - 2*float64(dots[j])
				x := rankedNode{leaf, float32(d)}
				if !have || x.before(best) {
					best, have = x, true
				}
			}
			evals += len(n.leaves)
		}
		w.best = w.best[:0]
		if have {
			w.best = append(w.best, best)
		}
		return w.best, evals
	}
	w.next = w.next[:0]
	for _, id := range w.holders {
		n := &t.nodes[id]
		scan(n, n.leaves)
	}
	w.best = selectNearest(w.best[:0], w.next, count)
	return w.best, evals
}

// TwoStageNearest is TwoStageAssign's search with the picked leaf ids,
// nearest first, and the number of centroid distance evaluations returned —
// the pairing Nearest and Evaluations give the beam search.
func (t Tree) TwoStageNearest(q []float32, top int) ([]int, int) {
	if t.root < 0 || top <= 0 || len(q) != t.dims {
		return nil, 0
	}
	var w treeWorkspace
	w.next = make([]rankedNode, 0, min(t.fanout, len(t.nodes)))
	best, evals := t.twoStageInto(&w, q, top, top)
	out := make([]int, len(best))
	for i, x := range best {
		out[i] = x.id
	}
	return out, evals
}

// twoStageInto runs the two-stage search into w: stage one ranks the root's
// children, selectNearest keeps the best top, stage two scans every leaf
// holder under each pick. Each stage's ranking is exact, so the only
// approximation is which subtrees stage two opens.
func (t Tree) twoStageInto(w *treeWorkspace, q []float32, top, count int) ([]rankedNode, int) {
	if t.root < 0 || top <= 0 || count <= 0 || len(q) != t.dims {
		return nil, 0
	}
	qq := Dot(q, q)
	root := &t.nodes[t.root]
	evals := 0
	w.next = w.next[:0]
	if len(root.children) == 0 {
		// A single-holder tree has no fan to rank: the holder is the one
		// bucket and stage two scans all of it.
		scanRank(t.dims, w, root, root.leaves, qq, q)
		evals += len(root.leaves)
	} else {
		scanRank(t.dims, w, root, root.children, qq, q)
		evals += len(root.children)
		// The picks stay in w.best so the workspace retains their buffer's
		// capacity between rows: assigning the stage-one result to a local
		// alone left w.best at stage two's count-sized cap, and stage one
		// re-grew it on every row.
		picks := selectNearest(w.best[:0], w.next, top)
		w.best = picks
		w.next = w.next[:0]
		for _, p := range picks {
			n := &t.nodes[p.id]
			if len(n.children) == 0 {
				scanRank(t.dims, w, n, n.leaves, qq, q)
				evals += len(n.leaves)
				continue
			}
			for _, h := range n.sub {
				hn := &t.nodes[h]
				scanRank(t.dims, w, hn, hn.leaves, qq, q)
				evals += len(hn.leaves)
			}
		}
	}
	w.best = selectNearest(w.best[:0], w.next, count)
	return w.best, evals
}

// scanRank is nearestInto's inner scan, factored out for the two-stage
// search: squared distances over a node's block by the norm identity, one
// dot product per candidate, float64 accumulation for near-tied ordering.
// nearestInto keeps its inline closure; this copy only reads the same
// fields.
func scanRank(dims int, w *treeWorkspace, n *treeNode, ids []int, qq float32, q []float32) {
	dots := w.dotsOf(q, n.block, len(ids))
	for j, id := range ids {
		d := float64(qq) + float64(n.norms[j]) - 2*float64(dots[j])
		w.next = append(w.next, rankedNode{id, float32(d)})
	}
}

// selectNearest returns the probe nearest candidates, nearest first, ties
// by id — exactly the head of a full sort by (d, id) — without sorting
// the rest. A beam search keeps sixteen of a few hundred at every level
// of every assignment, and the full sort was 80% of a merge's CPU at 10M.
// best is reused as the output buffer.
func selectNearest(best, cands []rankedNode, probe int) []rankedNode {
	for _, c := range cands {
		best = insertNearest(best, c, probe)
	}
	return best
}

// insertNearest offers one candidate to a sorted buffer of at most probe
// nodes. Folding candidates through it gives selectNearest's result.
func insertNearest(best []rankedNode, c rankedNode, probe int) []rankedNode {
	n := len(best)
	if n == probe && !c.before(best[n-1]) {
		return best
	}
	if n < probe {
		best = append(best, c)
		n++
	} else {
		best[n-1] = c
	}
	for i := n - 1; i > 0 && best[i].before(best[i-1]); i-- {
		best[i], best[i-1] = best[i-1], best[i]
	}
	return best
}

// before is the ranking order: nearer first, then lower id.
func (a rankedNode) before(b rankedNode) bool {
	return a.d < b.d || a.d == b.d && a.id < b.id
}

type rankedNode struct {
	id int
	d  float32
}

// Height returns the number of node levels, including the root and leaves.
func (t Tree) Height() int { return t.height }

// Dims is the width of every centroid in the tree.
func (t Tree) Dims() int { return t.dims }

// Leaves is the number of leaf centroids the tree ranks over.
func (t Tree) Leaves() int { return len(t.leaves) }

// Leaf returns leaf id's centroid, aliasing the tree's own storage: read
// only.
func (t Tree) Leaf(id int) []float32 { return t.leaves[id] }

// MarshalBinary encodes the hierarchy and leaf vectors in a versioned format.
func (t Tree) MarshalBinary() ([]byte, error) {
	var b bytes.Buffer
	b.WriteString("DVT\x01")
	for _, x := range []uint32{uint32(t.dims), uint32(t.fanout), uint32(t.root + 1), uint32(t.height), uint32(len(t.leaves)), uint32(len(t.nodes))} {
		binary.Write(&b, binary.LittleEndian, x)
	}
	writeVec := func(v []float32) {
		for _, x := range v {
			binary.Write(&b, binary.LittleEndian, math.Float32bits(x))
		}
	}
	for _, v := range t.leaves {
		if len(v) != t.dims {
			return nil, errors.New("ivf: mixed dimensions")
		}
		writeVec(v)
	}
	for _, n := range t.nodes {
		writeVec(n.centroid)
		binary.Write(&b, binary.LittleEndian, uint32(len(n.children)))
		binary.Write(&b, binary.LittleEndian, uint32(len(n.leaves)))
		for _, x := range n.children {
			binary.Write(&b, binary.LittleEndian, uint32(x))
		}
		for _, x := range n.leaves {
			binary.Write(&b, binary.LittleEndian, uint32(x))
		}
	}
	return b.Bytes(), nil
}

// UnmarshalTree decodes MarshalBinary output. Counts are checked against the
// bytes remaining before they size an allocation, and indices against the
// encoder's invariants: in range, the root is node zero, every child
// numbered after its parent (Build appends a node before recursing into its
// children, which is what makes traversal finite), every other node
// referenced by exactly one parent, every leaf owned by one node, and no
// node both branches and holds leaves.
func UnmarshalTree(data []byte) (Tree, error) {
	corrupt := errors.New("ivf: invalid tree encoding")
	if len(data) < 28 || string(data[:4]) != "DVT\x01" {
		return Tree{}, corrupt
	}
	r := bytes.NewReader(data[4:])
	var h [6]uint32
	for i := range h {
		if binary.Read(r, binary.LittleEndian, &h[i]) != nil {
			return Tree{}, errors.New("ivf: truncated tree")
		}
	}
	dims, nLeaves, nNodes := uint64(h[0]), uint64(h[4]), uint64(h[5])
	rem := uint64(r.Len())
	// Minimum footprint, independent of dims (zero-dimensional vectors are
	// free): every node carries two counts and every leaf the reference of
	// the node that owns it, plus 4*dims per vector.
	if fixed := 8*nNodes + 4*nLeaves; fixed > rem || nLeaves+nNodes > 0 && dims > (rem-fixed)/4/(nLeaves+nNodes) {
		return Tree{}, corrupt
	}
	root := int(h[2]) - 1
	if nNodes == 0 && (nLeaves != 0 || root != -1) || nNodes > 0 && root != 0 {
		return Tree{}, corrupt
	}
	t := Tree{dims: int(dims), fanout: int(h[1]), root: root, height: int(h[3]), leaves: make([][]float32, nLeaves), nodes: make([]treeNode, nNodes)}
	readVec := func() ([]float32, error) {
		v := make([]float32, t.dims)
		for i := range v {
			var x uint32
			if binary.Read(r, binary.LittleEndian, &x) != nil {
				return nil, errors.New("ivf: truncated tree")
			}
			v[i] = math.Float32frombits(x)
		}
		return v, nil
	}
	for i := range t.leaves {
		v, e := readVec()
		if e != nil {
			return Tree{}, e
		}
		t.leaves[i] = v
	}
	parented, owned := make([]bool, nNodes), make([]bool, nLeaves)
	unowned, unparented := nLeaves, nNodes // only the root may stay unparented
	for i := range t.nodes {
		v, e := readVec()
		if e != nil {
			return Tree{}, e
		}
		t.nodes[i].centroid = v
		var nc, nl uint32
		if binary.Read(r, binary.LittleEndian, &nc) != nil || binary.Read(r, binary.LittleEndian, &nl) != nil {
			return Tree{}, errors.New("ivf: truncated tree")
		}
		// A node is internal or a leaf holder, never both: nearest descends
		// children and ignores a mixed node's leaves, hiding them from search.
		if uint64(nc)+uint64(nl) > uint64(r.Len())/4 || nc > 0 && nl > 0 {
			return Tree{}, corrupt
		}
		t.nodes[i].children = make([]int, nc)
		t.nodes[i].leaves = make([]int, nl)
		for j := range t.nodes[i].children {
			var x uint32
			if binary.Read(r, binary.LittleEndian, &x) != nil {
				return Tree{}, errors.New("ivf: truncated tree")
			}
			if uint64(x) >= nNodes || int(x) <= i || parented[x] {
				return Tree{}, corrupt
			}
			parented[x] = true
			unparented--
			t.nodes[i].children[j] = int(x)
		}
		for j := range t.nodes[i].leaves {
			var x uint32
			if binary.Read(r, binary.LittleEndian, &x) != nil {
				return Tree{}, errors.New("ivf: truncated tree")
			}
			if uint64(x) >= nLeaves || owned[x] {
				return Tree{}, corrupt
			}
			owned[x] = true
			unowned--
			t.nodes[i].leaves[j] = int(x)
		}
	}
	if r.Len() != 0 || unowned != 0 || unparented != min(nNodes, 1) {
		return Tree{}, corrupt
	}
	t.index()
	return t, nil
}

// Clone returns a deep copy of the tree: every centroid, child and leaf
// list, scan block, and norm slice is duplicated, and the clone's leaves
// address the clone's blocks. Neither tree can observe the other's Upsert:
// a decoded tree sits in a shared cache behind concurrent searches, so a
// caller that means to mutate clones first and swaps the result in.
func (t Tree) Clone() Tree {
	c := Tree{
		dims: t.dims, fanout: t.fanout, root: t.root, height: t.height,
		nodes:  make([]treeNode, len(t.nodes)),
		leaves: make([][]float32, len(t.leaves)),
	}
	owned := make([]bool, len(t.leaves))
	for i := range t.nodes {
		n := &t.nodes[i]
		m := &c.nodes[i]
		m.centroid = append([]float32(nil), n.centroid...)
		m.children = append([]int(nil), n.children...)
		m.leaves = append([]int(nil), n.leaves...)
		m.block = append([]float32(nil), n.block...)
		m.norms = append([]float32(nil), n.norms...)
		m.sub = append([]int(nil), n.sub...)
		// An indexed holder's block slots are the leaves it owns, so the
		// clone's leaf references re-point at the clone's block. Blocks of
		// an unindexed tree (hand-built in tests) do not back their leaves;
		// those stay value-copied below.
		if c.dims > 0 && len(m.block) == len(m.leaves)*c.dims {
			for j, id := range m.leaves {
				c.leaves[id] = m.block[j*c.dims : (j+1)*c.dims : (j+1)*c.dims]
				owned[id] = true
			}
		}
	}
	for id, v := range c.leaves {
		if !owned[id] {
			c.leaves[id] = append([]float32(nil), v...)
		}
	}
	return c
}

// Upsert replaces leaf id's centroid with v, or appends v as a new leaf
// when id equals the number of leaves; any other id is an error, since
// appends must stay contiguous. The height never changes and the result
// round-trips through MarshalBinary and UnmarshalTree.
//
// A replace writes v into the slot the leaf's reference already aliases —
// index() made t.leaves[id] a sub-slice of the owning holder's block, so
// one copy serves both views — and refreshes that holder's norm. Internal
// centroids are deliberately not recomputed: a re-centred hood moves by a
// fraction of its radius, the beam walks a handful of candidates per
// level, and holders are ranked by their means, which drift negligibly
// over a run of upserts. Recentring is what the next full Build is for.
//
// An append descends the hierarchy the way Nearest does (beam = fanout) to
// find the holder whose centroid is nearest to v, appends the leaf there,
// and gives that holder a fresh block — holder blocks are sub-slices of
// the one leaf arena, so growing in place would trample a neighbouring
// hood — while re-pointing the holder's leaves at the new block. A holder
// is allowed to grow past the fanout: search scans every leaf of a visited
// holder, so correctness holds and quality degrades slowly; rebuild once
// enough upserts have accumulated.
func (t *Tree) Upsert(id int, v []float32) error {
	if len(t.nodes) == 0 && len(t.leaves) == 0 {
		// An empty tree adopts v's dimensions: Build over no centroids
		// leaves them unset.
		t.dims = len(v)
		t.fanout = max(t.fanout, 2)
		t.root = -1
	} else if len(v) != t.dims {
		return fmt.Errorf("ivf: upsert of a %d-dimensional vector into a %d-dimensional tree", len(v), t.dims)
	}
	if id < 0 {
		return fmt.Errorf("ivf: negative leaf id %d", id)
	}
	if id > len(t.leaves) {
		return fmt.Errorf("ivf: leaf id %d beyond %d leaves: appends must be contiguous", id, len(t.leaves))
	}
	if id < len(t.leaves) {
		t.replace(id, v)
		return nil
	}
	t.appendLeaf(v)
	return nil
}

// Truncate cuts the tree to its first n leaves: each later leaf leaves its
// holder, which gets a fresh block of the leaves it keeps (holder blocks
// are sub-slices of one arena, as in appendLeaf). Internal centroids stay,
// as they do under Upsert, and a holder may be left empty; the next Build
// rebalances.
func (t *Tree) Truncate(n int) error {
	if n < 0 || n > len(t.leaves) {
		return fmt.Errorf("ivf: truncate to %d of %d leaves", n, len(t.leaves))
	}
	if n == len(t.leaves) {
		return nil
	}
	for i := range t.nodes {
		h := &t.nodes[i]
		if len(h.children) > 0 || !slices.ContainsFunc(h.leaves, func(id int) bool { return id >= n }) {
			continue
		}
		var leaves []int
		var norms, blk []float32
		for j, id := range h.leaves {
			if id < n {
				leaves, norms = append(leaves, id), append(norms, h.norms[j])
				blk = append(blk, t.leaves[id]...)
			}
		}
		h.leaves, h.norms, h.block = leaves, norms, blk
		for j, id := range leaves {
			t.leaves[id] = blk[j*t.dims : (j+1)*t.dims : (j+1)*t.dims]
		}
	}
	clear(t.leaves[n:])
	t.leaves = t.leaves[:n]
	return nil
}

// replace writes v into the leaf's slot and refreshes the owning holder's
// norm. Finding the owner walks every holder's leaf list — O(leaves) int
// comparisons, noise next to the k-means and re-serialization a replace
// would otherwise force — and keeps the leaf's reference aliasing its
// block slot.
func (t *Tree) replace(id int, v []float32) {
	for i := range t.nodes {
		n := &t.nodes[i]
		if len(n.children) > 0 {
			continue // internal; a valid tree ranks leaves only in holders
		}
		for j, lid := range n.leaves {
			if lid == id {
				copy(t.leaves[id], v)
				n.norms[j] = Dot(v, v)
				return
			}
		}
	}
	copy(t.leaves[id], v) // unowned by any holder, so unreachable from search
}

// appendLeaf adds a leaf to the holder whose centroid is nearest to v. An
// empty tree becomes a single root holder.
func (t *Tree) appendLeaf(v []float32) {
	if len(t.nodes) == 0 {
		blk := append([]float32(nil), v...)
		t.nodes = []treeNode{{
			centroid: append([]float32(nil), v...),
			leaves:   []int{0},
			block:    blk,
			norms:    []float32{Dot(v, v)},
		}}
		t.root, t.height = 0, 1
		t.leaves = []([]float32){blk[:len(v):len(v)]}
		return
	}
	n := &t.nodes[t.nearestHolder(v)]
	old := n.block
	id := len(t.leaves)
	n.leaves = append(n.leaves, id)
	n.norms = append(n.norms, Dot(v, v))
	blk := make([]float32, 0, len(old)+len(v))
	blk = append(blk, old...)
	blk = append(blk, v...)
	n.block = blk
	for j, lid := range n.leaves[:len(n.leaves)-1] {
		t.leaves[lid] = blk[j*len(v) : (j+1)*len(v) : (j+1)*len(v)]
	}
	t.leaves = append(t.leaves, blk[len(old):len(old)+len(v):len(old)+len(v)])
}

// nearestHolder descends the hierarchy the way nearestInto does — same
// scan-by-norm-identity ranking, same selectNearest beam, beam = fanout —
// but keeps the visited leaf holders instead of their leaves, and returns
// the holder whose centroid is nearest to q (ties by lower id, -1 for an
// empty tree). Upsert appends place new leaves with it; nothing else
// exposes it.
func (t Tree) nearestHolder(q []float32) int {
	if t.root < 0 || len(q) != t.dims {
		return -1
	}
	beam := max(t.fanout, 2)
	var w treeWorkspace
	w.next = make([]rankedNode, 0, min(beam, len(t.nodes)))
	w.active = append(w.active[:0], t.root)
	qq := Dot(q, q)
	for len(w.active) > 0 {
		w.next = w.next[:0]
		for _, id := range w.active {
			n := &t.nodes[id]
			if len(n.children) == 0 {
				w.holders = append(w.holders, id)
				continue
			}
			for j, cid := range n.children {
				c := n.block[j*t.dims : (j+1)*t.dims]
				d := float64(qq) + float64(n.norms[j]) - 2*float64(Dot(q, c))
				w.next = append(w.next, rankedNode{cid, float32(d)})
			}
		}
		if len(w.next) == 0 {
			break
		}
		w.best = selectNearest(w.best[:0], w.next, beam)
		w.active = w.active[:0]
		for _, x := range w.best {
			w.active = append(w.active, x.id)
		}
	}
	bestID, bestD := -1, float32(math.Inf(1))
	for _, id := range w.holders {
		c := t.nodes[id].centroid
		d := float32(float64(qq) + float64(Dot(c, c)) - 2*float64(Dot(q, c)))
		if d < bestD || d == bestD && id < bestID { // nearer first, ties by id
			bestID, bestD = id, d
		}
	}
	return bestID
}
