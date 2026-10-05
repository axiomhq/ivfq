package ivfq

import (
	"fmt"
	"math/rand"
	"slices"
	"testing"
)

// filteredExact is the oracle: the probe nearest eligible leaves by the
// float32 distance the tree computes, ties by id.
func filteredExact(t Tree, q []float32, eligible []int, probe int) []int {
	qq := Dot(q, q)
	seen := map[int]bool{}
	var c []rankedNode
	for _, id := range eligible {
		if id < 0 || id >= t.Leaves() || seen[id] {
			continue
		}
		seen[id] = true
		v := t.Leaf(id)
		c = append(c, rankedNode{id, float32(float64(qq) + float64(Dot(v, v)) - 2*float64(Dot(q, v)))})
	}
	best := selectNearest(nil, c, probe)
	out := make([]int, len(best))
	for i, x := range best {
		out[i] = x.id
	}
	return out
}

func overlap(got, want []int) float64 {
	if len(want) == 0 {
		return 1
	}
	n := 0
	for _, x := range got {
		if slices.Contains(want, x) {
			n++
		}
	}
	return float64(n) / float64(len(want))
}

// When the pool covers every eligible leaf, FilteredNearest ranks exactly
// those leaves, nothing else, with no internal distance, float32 or int8.
func TestFilteredNearestCoversSmallSubset(t *testing.T) {
	for _, dims := range []int{16, 256} {
		v := treeData(3000, dims)
		for i := range 20 {
			v[500+i] = v[i] // ties by id
		}
		tree := mustBuild(v, 100)
		if dims >= 256 {
			tree.QuantizeI8()
		}
		r := rand.New(rand.NewSource(3))
		for _, n := range []int{1, 7, 64, 400} {
			eligible := r.Perm(len(v))[:n]
			eligible = append(eligible, eligible[0], -1, len(v)+5) // duplicates and junk
			for qi, q := range append(treeData(10, dims), v[eligible[0]]) {
				for _, probe := range []int{1, 5, 96} {
					got, work := tree.FilteredNearest(q, eligible, 16, probe)
					want := filteredExact(tree, q, eligible, probe)
					if !slices.Equal(got, want) {
						t.Fatalf("dims %d n %d query %d probe %d:\n got %v\nwant %v", dims, n, qi, probe, got, want)
					}
					if work.Total() > 2*n || dims < 256 && (work.Quantized != 0 || work.Float32 != n) || dims >= 256 && work.Quantized != n {
						t.Fatalf("dims %d n %d probe %d: work %+v, want only the %d eligible leaves", dims, n, probe, work, n)
					}
				}
			}
		}
		if got, work := tree.FilteredNearest(v[0], nil, 16, 5); got != nil || work.Total() != 0 {
			t.Fatalf("no eligible leaves: %v %+v", got, work)
		}
		if got, work := tree.FilteredNearest(v[0], []int{-3, len(v)}, 16, 5); got != nil || work.Total() != 0 {
			t.Fatalf("only junk eligible: %v %+v", got, work)
		}
	}
}

// On a 12,000-leaf tree (height 3 at fanout 100) a beam over eligible
// subtrees scores a bounded number of centroids, never an ineligible leaf
// or an internal node with nothing eligible beneath it, and ranks the
// probe nearest eligible leaves to high recall for rare, clustered, far
// clustered and broad filters, in float32 and through int8.
func TestFilteredNearestLargeTreeBounded(t *testing.T) {
	const leaves, probe = 12_000, 96
	beam := max(16, probe/8)
	for _, dims := range []int{16, 256} {
		t.Run(fmt.Sprint(dims), func(t *testing.T) {
			v := treeRandScaled(leaves, dims, 21, 2, 1)
			tree := mustBuild(v, 100)
			if tree.Height() < 3 {
				t.Fatalf("height %d: want a tree with internal levels below the root", tree.Height())
			}
			if dims >= 256 {
				tree.QuantizeI8()
			}
			internal := 0
			for _, n := range tree.nodes {
				internal += len(n.children)
			}
			qs := treeRandScaled(24, dims, 99, 2, 1)
			r := rand.New(rand.NewSource(5))
			// The leaves of one top-level subtree, and of the root's child
			// farthest from the first query: clustered filters, near and far.
			under := func(node int) []int {
				var out []int
				var walk func(int)
				walk = func(id int) {
					n := &tree.nodes[id]
					out = append(out, n.leaves...)
					for _, c := range n.children {
						walk(c)
					}
				}
				walk(node)
				return out
			}
			root := &tree.nodes[tree.root]
			far, farD := -1, float32(-1)
			for _, c := range root.children {
				if d := L2Sq(qs[0], tree.nodes[c].centroid); d > farD {
					far, farD = c, d
				}
			}
			// The root's children by distance from the first query, the
			// farthest 30 of them: a clustered filter large enough for the
			// beam, all of it far from the query.
			byFar := slices.Clone(root.children)
			slices.SortFunc(byFar, func(a, b int) int {
				da, db := L2Sq(qs[0], tree.nodes[a].centroid), L2Sq(qs[0], tree.nodes[b].centroid)
				switch {
				case da > db:
					return -1
				case da < db:
					return 1
				}
				return 0
			})
			var farMany []int
			for _, c := range byFar[:min(30, len(byFar))] {
				farMany = append(farMany, under(c)...)
			}
			// Asymmetric: 1% of the leaves anywhere, and every far leaf.
			asym := slices.Clone(farMany)
			for _, id := range r.Perm(leaves)[:leaves/100] {
				asym = append(asym, id)
			}
			filters := map[string][]int{
				"far-clusters-many": farMany,
				"asymmetric":        asym,
				"stripe-5": func() []int {
					var o []int
					for i := 0; i < leaves; i += 5 {
						o = append(o, i)
					}
					return o
				}(),
				"rare":         r.Perm(leaves)[:150],
				"near-cluster": under(root.children[0]),
				"far-cluster":  under(far),
				"broad-half":   r.Perm(leaves)[:leaves/2],
				"broad-90":     r.Perm(leaves)[:leaves*9/10],
				"stripe-sparse": func() []int { // 1 in 40, every subtree thinly eligible
					var o []int
					for i := 0; i < leaves; i += 40 {
						o = append(o, i)
					}
					return o
				}(),
			}
			for name, eligible := range filters {
				var recall float64
				maxWork := 0
				for qi, q := range qs {
					got, work := tree.FilteredNearest(q, eligible, beam, probe)
					want := filteredExact(tree, q, eligible, probe)
					for _, id := range got {
						if !slices.Contains(eligible, id) {
							t.Fatalf("%s query %d: ineligible leaf %d returned", name, qi, id)
						}
					}
					if len(got) != len(want) {
						t.Fatalf("%s query %d: %d leaves, want %d", name, qi, len(got), len(want))
					}
					recall += overlap(got, want)
					// Leaf work never exceeds the eligible leaves; internal work
					// never exceeds the internal nodes.
					leafWork := work.Quantized
					if dims < 256 {
						leafWork = work.Float32 - min(work.Float32, internal)
					}
					if leafWork > len(eligible) || work.Total() > internal+2*len(eligible) {
						t.Fatalf("%s query %d: work %+v, %d eligible, %d internal", name, qi, work, len(eligible), internal)
					}
					maxWork = max(maxWork, work.Total())
				}
				recall /= float64(len(qs))
				t.Logf("%s: %d eligible, recall@%d %.4f, max centroid work %d of %d leaves", name, len(eligible), probe, recall, maxWork, leaves)
				if recall < 0.95 {
					t.Fatalf("%s: recall %.4f", name, recall)
				}
				// Bounded: a broad filter's search costs about what the
				// unfiltered beam does, far below a full leaf ranking.
				if len(eligible) > max(2*probe, beam*100) && maxWork > leaves/2 {
					t.Fatalf("%s: %d centroid evaluations for %d leaves", name, maxWork, leaves)
				}
			}
		})
	}
}

// With every leaf eligible the filtered search is no worse than the
// unfiltered beam at the same beam.
func TestFilteredNearestAllEligible(t *testing.T) {
	v := treeRandScaled(12_000, 16, 31, 2, 1)
	tree := mustBuild(v, 100)
	all := make([]int, len(v))
	for i := range all {
		all[i] = i
	}
	var fr, ur float64
	for _, q := range treeRandScaled(16, 16, 77, 2, 1) {
		want := filteredExact(tree, q, all, 96)
		got, _ := tree.FilteredNearest(q, all, 16, 96)
		base, _ := tree.EvaluationsBeam(q, 16, 96)
		fr += overlap(got, want)
		ur += overlap(base, want)
	}
	if fr < ur {
		t.Fatalf("filtered recall %.3f below unfiltered %.3f", fr/16, ur/16)
	}
}

// EvaluationsBeamI8Work names EvaluationsBeamI8's leaves and float32
// count, and counts every int8 distance of the leaves it scanned.
func TestEvaluationsBeamI8Work(t *testing.T) {
	v := treeData(3000, 256)
	tree := mustBuild(v, 100)
	tree.QuantizeI8()
	for _, q := range treeData(8, 256) {
		ids, e := tree.EvaluationsBeamI8(q, 16, 96)
		ids2, work := tree.EvaluationsBeamI8Work(q, 16, 96)
		_, all := tree.EvaluationsBeam(q, 16, 96)
		if !slices.Equal(ids, ids2) || work.Float32 != e || work.Quantized == 0 {
			t.Fatalf("work %+v, EvaluationsBeamI8 %d", work, e)
		}
		// Every leaf the float32 search scanned was scored in int8.
		w := workspaces.Get().(*treeWorkspace)
		internal := tree.beamHolders(w, q, 16)
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
