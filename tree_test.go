package ivfq

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"reflect"
	"testing"
)

// treeRand draws n clustered vectors: every dimension sits at an offset
// shared by the vector's cluster plus unit noise, so clusters are tight
// and far apart — the shape the tree tests reason about. treeData is the
// canonical fixed-seed instance the existing tests share.
func treeRand(n, d int, seed int64) [][]float32 {
	r := rand.New(rand.NewSource(seed))
	v := make([][]float32, n)
	for i := range v {
		v[i] = make([]float32, d)
		c := float32(i%64) * 20
		for j := range v[i] {
			v[i][j] = c + float32(r.NormFloat64())
		}
	}
	return v
}

func treeData(n, d int) [][]float32 { return treeRand(n, d, 7) }

// treeRandScaled is treeRand's 64-cluster layout at a chosen cluster
// spacing and noise, centred on the origin.
func treeRandScaled(n, d int, seed int64, spacing, noise float32) [][]float32 {
	r := rand.New(rand.NewSource(seed))
	v := make([][]float32, n)
	for i := range v {
		v[i] = make([]float32, d)
		c := (float32(i%64) - 31.5) * spacing
		for j := range v[i] {
			v[i][j] = c + float32(r.NormFloat64())*noise
		}
	}
	return v
}

func TestTreeOracleAndEncoding(t *testing.T) {
	v := treeData(4096, 16)
	tree := mustBuild(v, 16)
	q := append([]float32(nil), v[1234]...)
	got, n := tree.Evaluations(q, 16)
	if !contains(got, Nearest(v, q)) {
		t.Fatalf("flat nearest absent: %v", got)
	}
	if n > tree.Height()*16*16+16*16 {
		t.Fatalf("evaluations %d exceed bound", n)
	}
	b, err := tree.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	copy, err := UnmarshalTree(b)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, copy.Nearest(q, 16)) {
		t.Fatal("encoding changed traversal")
	}
	if !reflect.DeepEqual(b, mustBytes(mustBuild(v, 16))) {
		t.Fatal("build is not deterministic")
	}
}

// TestTreeRankerMatchesNearest pins Ranker to Nearest: the same leaves in
// the same order for every probe width, with the pooled scratch reused
// across calls and shared by concurrent callers.
func TestTreeRankerMatchesNearest(t *testing.T) {
	v := treeData(4096, 16)
	tree := mustBuild(v, 16)
	for _, probe := range []int{1, 4, 16, 64} {
		rank := tree.Ranker(probe)
		done := make(chan error, 4)
		for g := range 4 {
			go func() {
				for i := g; i < len(v); i += 97 {
					var got []int
					rank(v[i], func(id int) { got = append(got, id) })
					if want := tree.Nearest(v[i], probe); !reflect.DeepEqual(got, want) {
						done <- fmt.Errorf("probe %d row %d: ranker %v, Nearest %v", probe, i, got, want)
						return
					}
				}
				done <- nil
			}()
		}
		for range 4 {
			if err := <-done; err != nil {
				t.Fatal(err)
			}
		}
	}
	if raceEnabled {
		return
	}
	rank := tree.Ranker(16)
	rank(v[5], func(int) {})
	if allocs := testing.AllocsPerRun(100, func() { rank(v[5], func(int) {}) }); allocs > 0 {
		t.Fatalf("a warm ranker allocates %.0f times per call", allocs)
	}
}

func contains(v []int, x int) bool {
	for _, y := range v {
		if x == y {
			return true
		}
	}
	return false
}
func mustBytes(t Tree) []byte {
	b, e := t.MarshalBinary()
	if e != nil {
		panic(e)
	}
	return b
}
func mustBuild(v [][]float32, fanout int) Tree {
	t, err := Build(context.Background(), v, fanout)
	if err != nil {
		panic(err)
	}
	return t
}

// nearestIntoReference is the original append-then-select beam search.
// Keep it independent of nearestInto so changes to the hot path have an
// exact traversal and evaluation-count oracle.
func (t Tree) nearestIntoReference(w *treeWorkspace, q []float32, beam, count int) ([]rankedNode, int) {
	if t.root < 0 || beam <= 0 || count <= 0 || len(q) != t.dims {
		return nil, 0
	}
	w.active = append(w.active[:0], t.root)
	w.holders = w.holders[:0]
	evals := 0
	qq := Dot(q, q)
	// scan ranks a node's block: squared distances by the norm identity,
	// which shares q's norm across the block and skips L2Sq's sqrt. The
	// three terms combine in float64: on long vectors they are ~1e7 and
	// the distance ~1e5, and float32 would lose the low bits that order
	// near-tied candidates.
	scan := func(n *treeNode, ids []int) {
		for j, id := range ids {
			c := n.block[j*t.dims : (j+1)*t.dims]
			d := float64(qq) + float64(n.norms[j]) - 2*float64(Dot(q, c))
			w.next = append(w.next, rankedNode{id, float32(d)})
		}
		evals += len(ids)
	}
	for len(w.active) > 0 {
		w.next = w.next[:0]
		for _, id := range w.active {
			n := &t.nodes[id]
			if len(n.children) == 0 {
				w.holders = append(w.holders, id)
				continue
			}
			scan(n, n.children)
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
	w.next = w.next[:0]
	for _, id := range w.holders {
		n := &t.nodes[id]
		scan(n, n.leaves)
	}
	w.best = selectNearest(w.best[:0], w.next, count)
	return w.best, evals
}

func TestNearestIntoMatchesReference(t *testing.T) {
	for _, fanout := range []int{4, 8, 100} {
		for _, k := range []int{1, 7, 100, 101, 1000, 5000} {
			for _, dims := range []int{8, 128} {
				name := fmt.Sprintf("k=%d/f=%d/d=%d", k, fanout, dims)
				t.Run(name, func(t *testing.T) {
					v := treeRand(k, dims, int64(k*1000+fanout*10+dims))
					if k > 1 {
						copy(v[k-1], v[0])
						if k > 3 {
							copy(v[k-2], v[1])
						}
					}
					tree := mustBuild(v, fanout)
					rows := append(treeRand(4, dims, int64(k+fanout+dims)), v[0], v[k/2], v[k-1])
					var gotW, wantW treeWorkspace
					for row, q := range rows {
						for _, count := range []int{1, 5, 16} {
							for _, beam := range []int{1, 4, 16} {
								got, gotEvals := tree.nearestInto(&gotW, q, beam, count)
								want, wantEvals := tree.nearestIntoReference(&wantW, q, beam, count)
								if gotEvals != wantEvals || !reflect.DeepEqual(got, want) {
									t.Fatalf("row=%d count=%d beam=%d: got (%v, %d), want (%v, %d)", row, count, beam, got, gotEvals, want, wantEvals)
								}
							}
						}
					}
				})
			}
		}
	}
}

func TestTreeRecall(t *testing.T) {
	v := treeData(10000, 32)
	tree := mustBuild(v, 32)
	r := rand.New(rand.NewSource(9))
	for _, p := range []int{8, 32, 128} {
		hits := 0
		for i := 0; i < 100; i++ {
			q := v[r.Intn(len(v))]
			if contains(tree.Nearest(q, p), Nearest(v, q)) {
				hits++
			}
		}
		t.Logf("recall@%d %.2f", p, float64(hits)/100)
		if hits < 95 {
			t.Fatalf("recall too low: %d", hits)
		}
	}
}

func TestTreeIdenticalCentroids(t *testing.T) {
	v := make([][]float32, 100)
	for i := range v {
		v[i] = []float32{1, 1}
	}
	if got := mustBuild(v, 4).Nearest(v[0], 8); len(got) != 8 {
		t.Fatalf("got %d leaves", len(got))
	}
}

// TestTreeTwoStage: the two-stage search is the exact flat scan whenever
// its stage-one top covers the whole root fan (the only approximation is
// which subtrees stage two opens), and its decoded round-trip behaves like
// the built tree's.
func TestTreeTwoStage(t *testing.T) {
	v := treeData(3000, 32)
	for _, tree := range []Tree{mustBuild(v, 16), mustRoundTrip(t, v, 16)} {
		fan := len(tree.nodes[tree.root].children)
		assign := tree.TwoStageAssigner(fan)
		for i := 0; i < 200; i++ {
			q := v[i]
			want := Nearest(v, q)
			if got := assign(q); got != want {
				t.Fatalf("top=%d: row %d: two-stage %d, flat %d", fan, i, got, want)
			}
			ids, evals := tree.TwoStageNearest(q, fan)
			if len(ids) == 0 || ids[0] != want {
				t.Fatalf("top=%d: row %d: nearest %v, flat %d", fan, i, ids, want)
			}
			// top = fan opens every subtree: stage one ranks each root
			// child once, stage two scans all leaves once.
			if evals != fan+len(v) {
				t.Fatalf("top=%d: row %d: evals %d, want %d", fan, i, evals, fan+len(v))
			}
			// top = 2 opens exactly the two subtrees whose centroids stage
			// one ranked best.
			if _, evals := tree.TwoStageNearest(q, 2); evals != fan+twoBucketSize(t, &tree, q) {
				t.Fatalf("top=2: row %d: evals %d, want %d", i, evals, fan+twoBucketSize(t, &tree, q))
			}
		}
	}
	// A single-holder tree has no fan: the two-stage scan is the flat scan.
	small := mustBuild(v[:30], 32)
	if got := small.TwoStageAssigner(4)(v[0]); got != Nearest(v[:30], v[0]) {
		t.Fatalf("single holder: two-stage %d, flat %d", got, Nearest(v[:30], v[0]))
	}
}

func mustRoundTrip(t *testing.T, v [][]float32, fanout int) Tree {
	t.Helper()
	tree, err := UnmarshalTree(mustBytes(mustBuild(v, fanout)))
	if err != nil {
		t.Fatal(err)
	}
	return tree
}

// twoBucketSize counts the leaves under the two root children whose
// centroids are nearest q — the subtrees a top-2 stage two opens. Ties
// break by child order, as selectNearest ranks them.
func twoBucketSize(t *testing.T, tree *Tree, q []float32) int {
	t.Helper()
	root := &tree.nodes[tree.root]
	type rank struct {
		d float32
		i int
	}
	rs := make([]rank, len(root.children))
	for j, c := range root.children {
		rs[j] = rank{L2Sq(tree.nodes[c].centroid, q), j}
	}
	for i := 1; i < len(rs); i++ {
		for k := i; k > 0 && rs[k].d < rs[k-1].d; k-- {
			rs[k], rs[k-1] = rs[k-1], rs[k]
		}
	}
	total := 0
	for _, r := range rs[:2] {
		n := &tree.nodes[root.children[r.i]]
		if len(n.children) == 0 {
			total += len(n.leaves)
			continue
		}
		for _, h := range n.sub {
			total += len(tree.nodes[h].leaves)
		}
	}
	return total
}

// TestUnmarshalTreeRejectsCorrupt: a corrupt encoding must come back as an
// error, never as an allocation bomb (counts), an index panic (out of
// range), an endless traversal (a child at or before its parent), or a
// silently disconnected hierarchy (root moved to a subtree, node unparented).
func TestUnmarshalTreeRejectsCorrupt(t *testing.T) {
	// Three 2-dim centroids at fanout 2: root 0 with children 1 and 2.
	good := mustBytes(mustBuild([][]float32{{0, 0}, {1, 0}, {0, 1}}, 2))
	if _, err := UnmarshalTree(good); err != nil {
		t.Fatal(err)
	}
	const hdr, vec = 28, 2 * 4
	rootChild := hdr + 3*vec + vec + 8 // node 0: centroid, two counts, children 1 and 2
	leaf := rootChild + 8 + vec + 8    // node 1: centroid, two counts, its first leaf
	cases := map[string]struct {
		off int
		val uint32
	}{
		"leaf count overflow": {hdr - 8, 1 << 31},
		"node count overflow": {hdr - 4, 1 << 31},
		"dims overflow":       {4, 1 << 30},
		"child out of range":  {rootChild, 99},
		"leaf out of range":   {leaf, 99},
		"cycle to root":       {rootChild, 0},
		"child claimed twice": {rootChild + 4, 1},
		"root is a subtree":   {12, 2}, // root header names node 1
	}
	for name, c := range cases {
		b := append([]byte(nil), good...)
		binary.LittleEndian.PutUint32(b[c.off:], c.val)
		tree, err := UnmarshalTree(b)
		if err == nil {
			t.Fatalf("%s: decoded %+v", name, tree)
		}
	}
	// Drop the root's second child reference (adjusting its child count) so
	// node 2 and its leaves are still encoded but unreachable from the root.
	b := append([]byte(nil), good[:rootChild+4]...)
	b = append(b, good[rootChild+8:]...)
	binary.LittleEndian.PutUint32(b[rootChild-8:], 1)
	if tree, err := UnmarshalTree(b); err == nil {
		t.Fatalf("unparented node: decoded %+v", tree)
	}
	// Zero dims make every vector free, so the leaf count must be bounded
	// by the references that own the leaves: this 36-byte encoding asks
	// for 2^32-1 leaves (~96 GiB of slice headers) with an 8-byte payload.
	zeroDims := []byte("DVT\x01")
	for _, x := range []uint32{0, 2, 1, 1, math.MaxUint32, 1} {
		zeroDims = binary.LittleEndian.AppendUint32(zeroDims, x)
	}
	zeroDims = append(zeroDims, make([]byte, 8)...)
	if tree, err := UnmarshalTree(zeroDims); err == nil {
		t.Fatalf("zero dims: decoded %d leaves", len(tree.leaves))
	}
}

// A node that both branches and holds leaves passes every ownership count,
// yet nearest descends its children and never ranks its leaves: a hidden
// part of the search space, so it is corrupt rather than merely odd.
func TestUnmarshalTreeRejectsMixedNode(t *testing.T) {
	mixed := Tree{dims: 2, fanout: 2, root: 0, height: 2, leaves: [][]float32{{0, 0}, {1, 1}}, nodes: []treeNode{
		{centroid: []float32{0.5, 0.5}, children: []int{1}, leaves: []int{0}},
		{centroid: []float32{1, 1}, leaves: []int{1}},
	}}
	b, err := mixed.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	if tree, err := UnmarshalTree(b); err == nil {
		t.Fatalf("mixed node: decoded %+v, whose search hides leaf 0: %v", tree, tree.Nearest([]float32{0, 0}, 2))
	}
}

// TestBuildCancelled: a cancelled context aborts the hierarchy build and
// returns its error instead of a tree — whether the build reaches k-means
// or is a single leaf that never would.
func TestBuildCancelled(t *testing.T) {
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := Build(cancelled, treeData(64, 4), 4); err != context.Canceled {
		t.Fatalf("Build on cancelled ctx: err = %v", err)
	}
	tree, err := Build(cancelled, [][]float32{{1}}, 2)
	if err != context.Canceled || !reflect.DeepEqual(tree, Tree{}) {
		t.Fatalf("leaf Build on cancelled ctx = (%+v, %v)", tree, err)
	}
}

func BenchmarkTreeNearest(b *testing.B) {
	for _, n := range []int{10000, 100000} {
		v := treeData(n, 128)
		q := v[n/2]
		tree := mustBuild(v, 100)
		b.Run(fmt.Sprintf("tree/%d", n), func(b *testing.B) {
			for b.Loop() {
				_ = tree.Nearest(q, 32)
			}
		})
		b.Run(fmt.Sprintf("flat/%d", n), func(b *testing.B) {
			for b.Loop() {
				_ = Nearest(v, q)
			}
		})
	}
}

func TestTreeAssignerMatchesNearest(t *testing.T) {
	for _, n := range []int{0, 1, 99, 100, 101, 2000} {
		t.Run(fmt.Sprint(n), func(t *testing.T) {
			tree, err := Build(context.Background(), treeData(n, 8), 100)
			if err != nil {
				t.Fatal(err)
			}
			assign := tree.Assigner(16)
			for _, q := range treeData(80, 8) {
				got := assign(q)
				want := tree.Nearest(q, 16)
				if len(want) == 0 {
					if got != -1 {
						t.Fatal(got)
					}
				} else if got != want[0] {
					t.Fatalf("assignment %d != query routing %d", got, want[0])
				}
			}
			if got := assign([]float32{1}); got != -1 {
				t.Fatalf("invalid dimensions: %d", got)
			}
		})
	}
}

func BenchmarkTreeAssigner(b *testing.B) {
	vectors := treeData(10000, 128)
	tree, err := Build(context.Background(), vectors, 100)
	if err != nil {
		b.Fatal(err)
	}
	for _, reuse := range []bool{false, true} {
		name := "nearest"
		if reuse {
			name = "assigner"
		}
		b.Run(name, func(b *testing.B) {
			assign := tree.Assigner(16)
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				q := vectors[i%len(vectors)]
				if reuse {
					_ = assign(q)
				} else {
					_ = tree.Nearest(q, 16)
				}
			}
		})
	}
}

// Byte-valued 128-dim vectors (the BIGANN shape) have norms near 1e7 and
// nearest-centroid distances near 1e5: the norm identity the tree ranks
// with must still agree with the exact scan on top-1 almost always and
// never lose the exact nearest from its beam.
func TestTreeAssignerAgreesWithFlatOnByteVectors(t *testing.T) {
	r := rand.New(rand.NewSource(11))
	bytes := func(n int) [][]float32 {
		out := make([][]float32, n)
		for i := range out {
			v := make([]float32, 128)
			for j := range v {
				c := float64((i%64)*4 + j%7*13)
				v[j] = float32(math.Round(math.Min(255, math.Max(0, c+r.NormFloat64()*12))))
			}
			out[i] = v
		}
		return out
	}
	centroids := bytes(4000)
	queries := bytes(400)
	tree := mustBuild(centroids, 100)
	assign := tree.Assigner(16)
	agree := 0
	for _, q := range queries {
		want := Nearest(centroids, q)
		if got := assign(q); got == want {
			agree++
		}
		if probes, _ := tree.Evaluations(q, 16); !contains(probes, want) {
			t.Fatalf("exact nearest %d missing from the beam", want)
		}
	}
	if agree < 396 {
		t.Fatalf("top-1 agreement %d/400 below 99%%", agree)
	}
}

// TestTreeUpsertReplaceAndAppend: 200 upserts — 100 in-place replaces, 100
// appends — keep the tree a faithful index of its current leaf set. Beam
// top-1 still agrees with the flat argmin, every appended vector finds
// itself, the height is untouched, and the mutated tree still round-trips
// through the encoding with identical search results.
func TestTreeUpsertReplaceAndAppend(t *testing.T) {
	n, d := 5000, 64
	// Clusters centred on the origin at unit scale, with a noise radius a
	// quarter of the cluster spacing: nearest-neighbour distances are then
	// within a few orders of magnitude of |q|^2, which is what keeps the
	// tree's float32 norm-identity ranking exact enough for a top-1 test
	// (treeRand's clusters, on a diagonal to 1,260, are not).
	const spacing, noise = 0.2, 0.05
	v := treeRandScaled(n, d, 7, spacing, noise)
	tree := mustBuild(v, 100)
	height := tree.Height()
	cur := append([][]float32(nil), v...) // the leaf set the flat scan ranks

	r := rand.New(rand.NewSource(13))
	var appended, replaced []int
	for i := 0; i < 200; i++ {
		if i%2 == 0 { // replace: perturb a random leaf in place
			id := r.Intn(n)
			q := append([]float32(nil), cur[id]...)
			for j := range q {
				q[j] += float32(r.NormFloat64()) * noise
			}
			if err := tree.Upsert(id, q); err != nil {
				t.Fatal(err)
			}
			cur[id] = q
			replaced = append(replaced, id)
			continue
		}
		// Append: a fresh centroid in its own cluster beyond the originals.
		nv := make([]float32, d)
		c := (float32(64+len(appended)) - 31.5) * spacing
		for j := range nv {
			nv[j] = c + float32(r.NormFloat64())*noise
		}
		if err := tree.Upsert(tree.Leaves(), nv); err != nil {
			t.Fatal(err)
		}
		appended = append(appended, tree.Leaves()-1)
		cur = append(cur, nv)
	}
	if tree.Leaves() != n+len(appended) || tree.Height() != height {
		t.Fatalf("leaves %d height %d: want %d leaves at height %d", tree.Leaves(), tree.Height(), n+len(appended), height)
	}

	// Rank the way callers do — Assigner descends with beam 16 — and
	// require top-1 agreement with the flat argmin over the current leaf
	// set. The tree ranks by the norm identity with float32 dot products,
	// so on this geometry (|q|^2 ~ 1e4 against nearest-neighbour distances
	// ~1e-2) two in-cluster candidates within a percent of each other can
	// swap; a pick that close is a tie, not a routing error. A wrong holder
	// or a stale slot shows up as a pick many times farther.
	assign := tree.Assigner(16)
	misses := 0
	for _, q := range treeRandScaled(500, d, 101, spacing, noise) {
		got, want := assign(q), Nearest(cur, q)
		if got != want && L2Sq(q, cur[got]) > L2Sq(q, cur[want])*1.02 {
			misses++
		}
	}
	if misses > 5 {
		t.Fatalf("top-1 agreement %d/500 below 99%%", 500-misses)
	}

	// Exact matches (distance zero) leave no room for rounding: every
	// appended leaf and every replaced slot is found by its own vector.
	for _, id := range appended {
		if got := assign(cur[id]); got != id {
			t.Fatalf("appended %d not found by a query equal to it: %d", id, got)
		}
	}
	for _, id := range replaced {
		if got := assign(cur[id]); got != id {
			t.Fatalf("replaced %d not found by a query equal to it: %d", id, got)
		}
	}

	b, err := tree.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := UnmarshalTree(b)
	if err != nil {
		t.Fatal(err)
	}
	if decoded.Leaves() != tree.Leaves() || decoded.Height() != tree.Height() {
		t.Fatalf("decoded %d leaves at height %d", decoded.Leaves(), decoded.Height())
	}
	for _, q := range treeRandScaled(100, d, 103, spacing, noise) {
		if !reflect.DeepEqual(tree.Nearest(q, 16), decoded.Nearest(q, 16)) {
			t.Fatal("round trip changed search results")
		}
	}
}

// TestTreeUpsertCloneIsolation: a clone shares no backing array, so an
// upsert on either of the pair — replace writes in place, append re-points
// leaf references — never shows in the other's leaf vectors or searches.
func TestTreeUpsertCloneIsolation(t *testing.T) {
	v := treeData(2000, 16)
	queries := treeRand(50, 16, 17)
	search := func(t Tree) [][]int {
		out := make([][]int, len(queries))
		for i, q := range queries {
			out[i] = t.Nearest(q, 8)
		}
		return out
	}
	leaves := func(t Tree) [][]float32 {
		out := make([][]float32, len(t.leaves))
		for i, x := range t.leaves {
			out[i] = append([]float32(nil), x...)
		}
		return out
	}
	upserts := func(tr *Tree) {
		q := append([]float32(nil), tr.leaves[7]...)
		q[0] += 3
		if err := tr.Upsert(7, q); err != nil {
			t.Fatal(err)
		}
		nv := make([]float32, 16)
		for j := range nv {
			nv[j] = float32(2000 + j)
		}
		if err := tr.Upsert(tr.Leaves(), nv); err != nil {
			t.Fatal(err)
		}
	}

	orig := mustBuild(v, 16)
	clone := orig.Clone()
	if !reflect.DeepEqual(orig, clone) {
		t.Fatal("clone differs from its original")
	}
	before, vectors := search(orig), leaves(orig)
	upserts(&clone)
	if !reflect.DeepEqual(orig.leaves, vectors) {
		t.Fatal("upsert on the clone touched the original's leaf vectors")
	}
	if !reflect.DeepEqual(search(orig), before) {
		t.Fatal("upsert on the clone changed the original's search results")
	}
	if clone.Leaves() != orig.Leaves()+1 {
		t.Fatalf("clone has %d leaves, original %d", clone.Leaves(), orig.Leaves())
	}

	// And the other way round, on a fresh pair.
	orig2 := mustBuild(v, 16)
	clone2 := orig2.Clone()
	before2, vectors2 := search(clone2), leaves(clone2)
	upserts(&orig2)
	if !reflect.DeepEqual(clone2.leaves, vectors2) {
		t.Fatal("upsert on the original touched the clone's leaf vectors")
	}
	if !reflect.DeepEqual(search(clone2), before2) {
		t.Fatal("upsert on the original changed the clone's search results")
	}
}

func TestTreeUpsertErrors(t *testing.T) {
	tree := mustBuild(treeData(100, 8), 8)
	if err := tree.Upsert(3, make([]float32, 7)); err == nil {
		t.Fatal("a vector of the wrong width was accepted")
	}
	if err := tree.Upsert(tree.Leaves()+1, make([]float32, 8)); err == nil {
		t.Fatal("a non-contiguous append was accepted")
	}
}

// TestTreeUpsertEmptyTree: an empty tree adopts the first upserted vector's
// dimensions and grows into a one-holder hierarchy, whether it came from
// Build over no centroids or is the zero value.
func TestTreeUpsertEmptyTree(t *testing.T) {
	empty, err := Build(context.Background(), nil, 8)
	if err != nil {
		t.Fatal(err)
	}
	v := []float32{1, 2, 3}
	if err := empty.Upsert(0, v); err != nil {
		t.Fatal(err)
	}
	if got := empty.Nearest(v, 1); !reflect.DeepEqual(got, []int{0}) {
		t.Fatalf("nearest on a one-leaf tree: %v", got)
	}
	if err := empty.Upsert(1, []float32{3, 2, 1}); err != nil {
		t.Fatal(err)
	}
	if empty.Leaves() != 2 || empty.Height() != 1 {
		t.Fatalf("leaves %d height %d after two appends", empty.Leaves(), empty.Height())
	}
	if got := empty.Nearest([]float32{3, 2, 1}, 1); !reflect.DeepEqual(got, []int{1}) {
		t.Fatalf("second leaf not nearest to itself: %v", got)
	}

	var zero Tree
	if err := zero.Upsert(0, v); err != nil {
		t.Fatal(err)
	}
	if got := zero.Nearest(v, 1); !reflect.DeepEqual(got, []int{0}) {
		t.Fatalf("zero-value tree: %v", got)
	}
}
