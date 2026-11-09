package ivfq

import (
	"bytes"
	"cmp"
	"context"
	"encoding/binary"
	"fmt"
	"github.com/axiomhq/ivfq/kmeans"
	"math"
	"math/rand"
	"reflect"
	"slices"
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
	got, n := tree.EvaluationsBeam(q, 16, 16)
	if !slices.Contains(got, kmeans.Nearest(v, q)) {
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

// TestTreeSearcherMatchesNearest pins Searcher to Nearest: the same leaves in
// the same order for every probe width, with each goroutine's scratch
// reused across calls.
func TestTreeSearcherMatchesNearest(t *testing.T) {
	v := treeData(4096, 16)
	tree := mustBuild(v, 16)
	for _, probe := range []int{1, 4, 16, 64} {
		done := make(chan error, 4)
		for g := range 4 {
			go func() {
				s := tree.NewSearcher()
				for i := g; i < len(v); i += 97 {
					got := s.Nearest(v[i], probe)
					if want := tree.Nearest(v[i], probe); !reflect.DeepEqual(got, want) {
						done <- fmt.Errorf("probe %d row %d: searcher %v, Nearest %v", probe, i, got, want)
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
	s := tree.NewSearcher()
	s.Nearest(v[5], 16)
	if allocs := testing.AllocsPerRun(100, func() { s.Nearest(v[5], 16) }); allocs > 0 {
		t.Fatalf("a warm searcher allocates %.0f times per call", allocs)
	}
}

func mustBytes(t FloatTree) []byte {
	b, e := t.MarshalBinary()
	if e != nil {
		panic(e)
	}
	return b
}

func mustBuild(v [][]float32, fanout int) FloatTree {
	t, err := Build(context.Background(), v, fanout)
	if err != nil {
		panic(err)
	}
	return t
}

// nearestIntoReference is the original append-then-select beam search.
// Keep it independent of nearestInto so changes to the hot path have an
// exact traversal and evaluation-count oracle.
func (t FloatTree) nearestIntoReference(w *treeWorkspace, q []float32, beam, count int) ([]rankedNode, int) {
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
			if slices.Contains(tree.Nearest(q, p), kmeans.Nearest(v, q)) {
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
	for _, tree := range []FloatTree{mustBuild(v, 16), mustRoundTrip(t, v, 16)} {
		fan := len(tree.nodes[tree.root].children)
		s := tree.NewSearcher()
		for i := 0; i < 200; i++ {
			q := v[i]
			want := kmeans.Nearest(v, q)
			if got := s.TwoStageAssign(q, fan); got != want {
				t.Fatalf("top=%d: row %d: two-stage %d, flat %d", fan, i, got, want)
			}
			var w treeWorkspace
			best, evals := tree.twoStageInto(&w, q, fan, fan)
			if ids := idsOf(best); len(ids) == 0 || ids[0] != want {
				t.Fatalf("top=%d: row %d: nearest %v, flat %d", fan, i, ids, want)
			}
			// top = fan opens every subtree: stage one ranks each root
			// child once, stage two scans all leaves once.
			if evals != fan+len(v) {
				t.Fatalf("top=%d: row %d: evals %d, want %d", fan, i, evals, fan+len(v))
			}
			// top = 2 opens exactly the two subtrees whose centroids stage
			// one ranked best.
			if _, evals := tree.twoStageInto(&w, q, 2, 2); evals != fan+twoBucketSize(t, &tree, q) {
				t.Fatalf("top=2: row %d: evals %d, want %d", i, evals, fan+twoBucketSize(t, &tree, q))
			}
		}
	}
	// A single-holder tree has no fan: the two-stage scan is the flat scan.
	small := mustBuild(v[:30], 32)
	if got := small.NewSearcher().TwoStageAssign(v[0], 4); got != kmeans.Nearest(v[:30], v[0]) {
		t.Fatalf("single holder: two-stage %d, flat %d", got, kmeans.Nearest(v[:30], v[0]))
	}
}

func mustRoundTrip(t *testing.T, v [][]float32, fanout int) FloatTree {
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
func twoBucketSize(t *testing.T, tree *FloatTree, q []float32) int {
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
	slices.SortStableFunc(rs, func(a, b rank) int { return cmp.Compare(a.d, b.d) })
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
		"fanout past int32":   {8, 1 << 31}, // negative on 32-bit
		"NaN leaf":            {hdr, 0x7fc00000},
		"NaN node":            {hdr + 3*vec, 0x7fc00000},
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
	mixed := FloatTree{dims: 2, fanout: 2, root: 0, height: 2, leaves: [][]float32{{0, 0}, {1, 1}}, nodes: []treeNode{
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
	if err != context.Canceled || !reflect.DeepEqual(tree, FloatTree{}) {
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
				_ = kmeans.Nearest(v, q)
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
			s := tree.NewSearcher()
			for _, q := range treeData(80, 8) {
				got := s.Assign(q, 16)
				want := tree.Nearest(q, 16)
				if len(want) == 0 {
					if got != -1 {
						t.Fatal(got)
					}
				} else if got != want[0] {
					t.Fatalf("assignment %d != query routing %d", got, want[0])
				}
			}
			if got := s.Assign([]float32{1}, 16); got != -1 {
				t.Fatalf("invalid dimensions: %d", got)
			}
		})
	}
}

func BenchmarkTreeSearcher(b *testing.B) {
	vectors := treeData(10000, 128)
	tree, err := Build(context.Background(), vectors, 100)
	if err != nil {
		b.Fatal(err)
	}
	for _, reuse := range []bool{false, true} {
		name := "nearest"
		if reuse {
			name = "searcher"
		}
		b.Run(name, func(b *testing.B) {
			s := tree.NewSearcher()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				q := vectors[i%len(vectors)]
				if reuse {
					_ = s.Assign(q, 16)
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
	s := tree.NewSearcher()
	agree := 0
	for _, q := range queries {
		want := kmeans.Nearest(centroids, q)
		if got := s.Assign(q, 16); got == want {
			agree++
		}
		if probes, _ := tree.EvaluationsBeam(q, 16, 16); !slices.Contains(probes, want) {
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
	s := tree.NewSearcher()
	misses := 0
	for _, q := range treeRandScaled(500, d, 101, spacing, noise) {
		got, want := s.Assign(q, 16), kmeans.Nearest(cur, q)
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
		if got := s.Assign(cur[id], 16); got != id {
			t.Fatalf("appended %d not found by a query equal to it: %d", id, got)
		}
	}
	for _, id := range replaced {
		if got := s.Assign(cur[id], 16); got != id {
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
	search := func(t FloatTree) [][]int {
		out := make([][]int, len(queries))
		for i, q := range queries {
			out[i] = t.Nearest(q, 8)
		}
		return out
	}
	leaves := func(t FloatTree) [][]float32 {
		out := make([][]float32, len(t.leaves))
		for i, x := range t.leaves {
			out[i] = append([]float32(nil), x...)
		}
		return out
	}
	upserts := func(tr *FloatTree) {
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
	before, _ := tree.MarshalBinary()
	for _, x := range []float32{float32(math.NaN()), float32(math.Inf(-1)), 3e19} {
		for _, id := range []int{3, tree.Leaves()} { // replace, append
			v := make([]float32, 8)
			v[5] = x
			if err := tree.Upsert(id, v); err == nil {
				t.Fatalf("Upsert(%d) of a %v coordinate was accepted", id, x)
			}
		}
	}
	if after, _ := tree.MarshalBinary(); !bytes.Equal(before, after) {
		t.Fatal("a rejected upsert changed the tree")
	}
	var empty FloatTree
	if err := empty.Upsert(0, []float32{float32(math.NaN()), 1}); err == nil || empty.Dims() != 0 {
		t.Fatalf("an empty tree took a rejected vector: err %v, dims %d", err, empty.Dims())
	}
	// Build holds its centroids to Upsert's rule, and to one width.
	for name, cs := range map[string][][]float32{
		"NaN":    {{0, 0}, {float32(math.NaN()), 1}, {1, 1}},
		"3e19":   {{0, 0}, {3e19, 1}, {1, 1}},
		"ragged": {{0, 0}, {1}, {1, 1}},
	} {
		if _, err := Build(context.Background(), cs, 2); err == nil {
			t.Fatalf("Build accepted a %s centroid", name)
		}
	}
	// Finite centroids whose distance to a finite vector overflows float32
	// still leave the vector a holder to join.
	far := mustBuild([][]float32{{1.5e19, 0}, {0, 1.5e19}}, 2)
	if err := far.Upsert(2, []float32{-1.5e19, 0}); err != nil {
		t.Fatal(err)
	}
	if _, err := UnmarshalTree(mustBytes(far)); err != nil {
		t.Fatalf("tree after an overflowing append: %v", err)
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

	// The zero value and a zero-dimensional tree search and round-trip.
	var none FloatTree
	if got := none.Nearest(nil, 4); len(got) != 0 {
		t.Fatalf("zero-value tree found %v", got)
	}
	if _, err := UnmarshalTree(mustBytes(none)); err != nil {
		t.Fatalf("zero-value tree does not round-trip: %v", err)
	}
	flat := mustBuild([][]float32{{}, {}, {}}, 2)
	if got := flat.Nearest([]float32{}, 2); len(got) != 2 {
		t.Fatalf("zero-dimensional tree found %v", got)
	}

	var zero FloatTree
	if err := zero.Upsert(0, v); err != nil {
		t.Fatal(err)
	}
	if got := zero.Nearest(v, 1); !reflect.DeepEqual(got, []int{0}) {
		t.Fatalf("zero-value tree: %v", got)
	}
}

// TestSelectNearestMatchesSort: the bounded selection is exactly the head
// of a full sort by (distance, id), for every probe including ones larger
// than the candidate set, with duplicate distances to exercise the tie.
func TestSelectNearestMatchesSort(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for trial := 0; trial < 200; trial++ {
		n := rng.Intn(600)
		cands := make([]rankedNode, n)
		for i := range cands {
			cands[i] = rankedNode{id: rng.Intn(n + 1), d: float32(rng.Intn(50)) / 7}
		}
		probe := 1 + rng.Intn(40)
		if trial%2 == 1 { // a beam in the hundreds, past n as often as not
			probe = 1 + rng.Intn(700)
		}
		want := slices.Clone(cands)
		slices.SortFunc(want, rankedNode.compare)
		want = want[:min(probe, len(want))]
		got := selectNearest(make([]rankedNode, 0, probe), cands, probe)
		if !slices.Equal(got, want) {
			t.Fatalf("trial %d (n=%d probe=%d): got %v want %v", trial, n, probe, got, want)
		}
	}
}

// The three benchmarks share one 10,000 × 128 tree, the shape the other
// tree benchmarks measure: Build is the cost an upsert-heavy caller
// defers, Clone is what it pays to mutate a cached tree safely, and an
// Upsert append is the incremental path those two exist to avoid.

func BenchmarkTreeBuild(b *testing.B) {
	v := treeData(10000, 128)
	b.ReportAllocs()
	for b.Loop() {
		if _, err := Build(context.Background(), v, 100); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkTreeUpsertAppend(b *testing.B) {
	v := treeData(10000, 128)
	base := mustBuild(v, 100)
	tree := base.Clone()
	pool := make([][]float32, 1024)
	r := rand.New(rand.NewSource(5))
	for i := range pool {
		nv := make([]float32, 128)
		for j := range nv {
			nv[j] = float32(r.NormFloat64())
		}
		pool[i] = nv
	}
	// Appends grow holders without bound, so every burst of 1,024 restarts
	// from a fresh clone: steady-state append cost, with the rare clone
	// amortized into ~0.1% of the measured operations.
	const burst = 1024
	done := 0
	b.ReportAllocs()
	for b.Loop() {
		if done == burst {
			tree = base.Clone()
			done = 0
		}
		if err := tree.Upsert(tree.Leaves(), pool[done]); err != nil {
			b.Fatal(err)
		}
		done++
	}
}

func BenchmarkTreeClone(b *testing.B) {
	v := treeData(10000, 128)
	base := mustBuild(v, 100)
	b.ReportAllocs()
	for b.Loop() {
		_ = base.Clone()
	}
}

// EvaluationsBeam at beam = probe is Nearest. Narrower, it evaluates
// fewer centroids and still returns probe leaves, nearest first by their
// exact distance, ties by id.
func TestEvaluationsBeam(t *testing.T) {
	v := treeData(5000, 16)
	tree := mustBuild(v, 100)
	for qi, q := range treeData(30, 16) {
		const probe = 96
		want := tree.Nearest(q, probe)
		got, gotEvals := tree.EvaluationsBeam(q, probe, probe)
		if !slices.Equal(got, want) {
			t.Fatalf("query %d: beam = probe gave %v, Nearest %v", qi, got, want)
		}
		narrow, evals := tree.EvaluationsBeam(q, 24, probe)
		if len(narrow) != probe || evals >= gotEvals {
			t.Fatalf("query %d: beam 24 gave %d leaves in %d evals, beam %d took %d", qi, len(narrow), evals, probe, gotEvals)
		}
		d := func(id int) float64 {
			return float64(Dot(q, q)) + float64(Dot(v[id], v[id])) - 2*float64(Dot(q, v[id]))
		}
		for i := 1; i < len(narrow); i++ {
			a, b := d(narrow[i-1]), d(narrow[i])
			if float32(a) > float32(b)+1e-3 {
				t.Fatalf("query %d: leaf %d at %v ranked before leaf %d at %v", qi, narrow[i-1], a, narrow[i], b)
			}
		}
	}
}

// EvaluationsBeamI8Work is EvaluationsBeam, leaf for leaf and in order, on
// every query: widths with an int8 tail, beams from narrow to every
// holder, a probe past the tree, duplicated leaves (ties by id), a query
// on a leaf and the zero query; after an Upsert (the int8 copies dropped)
// and on a Clone (never copied). It computes far fewer float32 distances
// than the leaves it scans.
func TestEvaluationsBeamI8IsEvaluationsBeam(t *testing.T) {
	for _, dims := range []int{16, 100, 128, 768} {
		v := treeData(3000, dims)
		for i := 0; i < 30; i++ {
			v[200+i] = v[i]
		}
		tree := mustBuild(v, 100)
		tree.QuantizeI8()
		qs := append(treeData(60, dims), v[7], make([]float32, dims))
		var exact, scanned int
		check := func(name string, tr FloatTree) {
			for qi, q := range qs {
				for _, c := range [][2]int{{16, 96}, {4, 50}, {200, 400}, {16, 1}, {16, 5000}} {
					want, all := tr.EvaluationsBeam(q, c[0], c[1])
					got, w := tr.EvaluationsBeamI8Work(q, c[0], c[1])
					e := w.Float32
					if !slices.Equal(got, want) {
						t.Fatalf("%s dims %d query %d beam %d probe %d:\n got %v\nwant %v", name, dims, qi, c[0], c[1], got, want)
					}
					exact, scanned = exact+e, scanned+all
				}
			}
		}
		check("quantized", tree)
		if exact*3 > scanned {
			t.Fatalf("dims %d: %d float32 distances for %d scanned", dims, exact, scanned)
		}
		check("clone", tree.Clone())
		changed := tree.Clone()
		changed.QuantizeI8()
		if err := changed.Upsert(3, v[2500]); err != nil {
			t.Fatal(err)
		}
		check("after upsert", changed)
	}
}

func BenchmarkTreeI8(b *testing.B) {
	for _, dims := range []int{128, 768} {
		v := treeData(4100, dims)
		tree := mustBuild(v, 100)
		tree.QuantizeI8()
		qs := treeData(64, dims)
		b.Run(fmt.Sprintf("f32/%d", dims), func(b *testing.B) {
			i := 0
			for b.Loop() {
				tree.EvaluationsBeam(qs[i%64], 16, 128)
				i++
			}
		})
		b.Run(fmt.Sprintf("i8/%d", dims), func(b *testing.B) {
			i, e := 0, 0
			for b.Loop() {
				_, w := tree.EvaluationsBeamI8Work(qs[i%64], 16, 128)
				e += w.Float32
				i++
			}
			b.ReportMetric(float64(e)/float64(i), "f32evals/op")
		})
	}
}

// EvaluationsBeamI8Work names EvaluationsBeam's leaves, counts no more
// float32 distances, and counts every int8 distance of the leaves it scanned.
func TestEvaluationsBeamI8Work(t *testing.T) {
	v := treeData(3000, 256)
	tree := mustBuild(v, 100)
	tree.QuantizeI8()
	for _, q := range treeData(8, 256) {
		ids, work := tree.EvaluationsBeamI8Work(q, 16, 96)
		want, all := tree.EvaluationsBeam(q, 16, 96)
		if !slices.Equal(ids, want) || work.Float32 > all || work.Quantized == 0 {
			t.Fatalf("work %+v, EvaluationsBeam %d", work, all)
		}
		// Every leaf the float32 search scanned was scored in int8.
		w := workspaces.Get().(*treeWorkspace)
		internal := tree.beamHolders(w, q, Dot(q, q), 16)
		workspaces.Put(w)
		if work.Quantized != all-internal {
			t.Fatalf("quantized %d, want %d leaves", work.Quantized, all-internal)
		}
	}
	plain := mustBuild(v, 100)
	for _, q := range treeData(4, 256) {
		_, e := plain.EvaluationsBeam(q, 16, 96)
		if _, work := plain.EvaluationsBeamI8Work(q, 16, 96); work.Quantized != 0 || work.Float32 != e {
			t.Fatalf("unquantized: %+v, want float32 %d", work, e)
		}
	}
}

// A chain of internal nodes makes index keep O(depth²) descendant lists, so a
// small blob must not be able to ask for an arbitrarily deep one.
func TestUnmarshalTreeRejectsDeepChain(t *testing.T) {
	const links = 4 * maxTreeDepth
	b := []byte("DVT\x01")
	for _, x := range []uint32{1, 2, 1, links + 1, 0, 2*links + 1} {
		b = binary.LittleEndian.AppendUint32(b, x)
	}
	node := func(children ...uint32) {
		b = binary.LittleEndian.AppendUint32(b, 0) // 1-dim centroid
		b = binary.LittleEndian.AppendUint32(b, uint32(len(children)))
		b = binary.LittleEndian.AppendUint32(b, 0)
		for _, c := range children {
			b = binary.LittleEndian.AppendUint32(b, c)
		}
	}
	for i := uint32(0); i < links; i++ {
		node(2*i+1, 2*i+2) // internal: an empty holder and the next link
		node()
	}
	node()
	if _, err := UnmarshalTree(b); err == nil {
		t.Fatal("decoded a chain deeper than maxTreeDepth")
	}
}

// FuzzUnmarshalTree: arbitrary bytes must decode or fail without panicking,
// and what decodes must search and re-encode to a tree that decodes again.
func FuzzUnmarshalTree(f *testing.F) {
	f.Add(mustBytes(mustBuild([][]float32{{0, 0}, {1, 0}, {0, 1}, {1, 1}, {2, 2}}, 2)))
	f.Add([]byte("DVT\x01"))
	f.Fuzz(func(t *testing.T, data []byte) {
		tree, err := UnmarshalTree(data)
		if err != nil {
			return
		}
		tree.Nearest(make([]float32, tree.Dims()), 2)
		b, err := tree.MarshalBinary()
		if err != nil {
			t.Fatal(err)
		}
		if _, err := UnmarshalTree(b); err != nil {
			t.Fatalf("re-encoded tree does not decode: %v", err)
		}
	})
}
