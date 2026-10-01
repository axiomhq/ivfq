package ivfq

import (
	"bytes"
	"context"
	"errors"
	"github.com/axiomhq/ivfq/kmeans"
	"math/rand"
	"reflect"
	"slices"
	"testing"
)

func TestCentroidsRoundTrip(t *testing.T) {
	cs := [][]float32{{1, 2}, {3, 4}, {5, 6}}
	got, err := DecodeCentroids(EncodeCentroids(cs), 2)
	if err != nil || !reflect.DeepEqual(got, cs) {
		t.Fatalf("round trip: %+v, %v", got, err)
	}
	if got, err := DecodeCentroids(EncodeCentroids(nil), 2); err != nil || len(got) != 0 {
		t.Fatalf("empty: %+v, %v", got, err)
	}
	if _, err := DecodeCentroids([]byte{1, 2, 3}, 2); !errors.Is(err, ErrCorrupt) {
		t.Fatalf("truncated centroids: %v, want ErrCorrupt", err)
	}
	// A lying count prefix must error fast, never OOM on pre-allocation.
	if _, err := DecodeCentroids([]byte{0xFF, 0xFF, 0xFF, 0xFF, 1, 2}, 2); !errors.Is(err, ErrCorrupt) {
		t.Fatalf("allocation-bomb centroids prefix: %v, want ErrCorrupt", err)
	}
	if _, err := DecodeCentroids(append(EncodeCentroids(cs), 0), 2); !errors.Is(err, ErrCorrupt) {
		t.Fatalf("trailing byte: %v, want ErrCorrupt", err)
	}
}

// FuzzDecodeCentroids leaves dims free down to 0: the decoder, not the
// caller, must reject dims <= 0, where a lying count prefix would spin
// 2^32 zero-byte reads.
func FuzzDecodeCentroids(f *testing.F) {
	valid := EncodeCentroids([][]float32{{1, 2}, {3, 4}})
	f.Add(valid, uint8(2))
	for _, s := range []struct {
		b []byte
		d uint8
	}{
		{nil, 2}, {[]byte{}, 2}, {[]byte{1, 2, 3}, 2}, // truncated count
		{[]byte{0, 0, 0, 0}, 2},                   // zero centroids
		{[]byte{0xFF, 0xFF, 0xFF, 0xFF, 1, 2}, 2}, // allocation-bomb prefix
		{[]byte{0xFF, 0xFF, 0xFF, 0x7F}, 1},       // huge count, no payload
		{append(valid, 0xFF), 2},                  // trailing garbage
		{valid, 1}, {valid, 16},                   // dims disagreement
		{bytes.Repeat([]byte{0xFF}, 128), 4},
	} {
		f.Add(s.b, s.d)
	}
	f.Fuzz(func(t *testing.T, data []byte, dimsB uint8) {
		dims := int(dimsB) % 17 // 0 included: the guard, not the caller, must hold
		got, err := DecodeCentroids(data, dims)
		if err != nil {
			if !errors.Is(err, ErrCorrupt) {
				t.Fatalf("error does not wrap ErrCorrupt: %v", err)
			}
			return
		}
		if dims <= 0 {
			t.Fatalf("DecodeCentroids accepted dims=%d", dims)
		}
		// Every centroid must be fully read: 4 bytes of count plus
		// dims*4 per entry can never exceed the input.
		if 4+len(got)*dims*4 > len(data) {
			t.Fatalf("%d centroids of %d dims decoded from %d bytes", len(got), dims, len(data))
		}
		for _, c := range got {
			if len(c) != dims {
				t.Fatalf("centroid has %d dims, want %d", len(c), dims)
			}
		}
	})
}

// TestDecodeRejectsZeroDims: at dims == 0 every entry is zero-length, so a
// lying count prefix would spin instead of dying on a truncated read.
func TestDecodeRejectsZeroDims(t *testing.T) {
	bomb := []byte{0xFF, 0xFF, 0xFF, 0xFF}
	if _, err := DecodeCentroids(bomb, 0); !errors.Is(err, ErrCorrupt) {
		t.Fatalf("DecodeCentroids at dims=0: %v", err)
	}
}

func TestCentroidDeltaRoundTripAndApply(t *testing.T) {
	base := [][]float32{{1, 2}, {3, 4}, {5, 6}}
	d1 := []CentroidUpsert{{ID: 1, Vec: []float32{30, 40}}, {ID: 3, Vec: []float32{7, 8}}}
	d2 := []CentroidUpsert{{ID: 4, Vec: []float32{9, 10}}}
	for _, d := range [][]CentroidUpsert{d1, d2, nil} {
		got, err := DecodeCentroidDelta(EncodeCentroidDelta(d), 2)
		if err != nil {
			t.Fatal(err)
		}
		if len(d) == 0 && len(got) == 0 {
			continue
		}
		if !reflect.DeepEqual(got, d) {
			t.Fatalf("round trip: got %v want %v", got, d)
		}
	}
	got, err := ApplyCentroidDeltas(base, d1, d2)
	if err != nil {
		t.Fatal(err)
	}
	want := [][]float32{{1, 2}, {30, 40}, {5, 6}, {7, 8}, {9, 10}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("applied: got %v want %v", got, want)
	}
	if !reflect.DeepEqual(base[1], []float32{3, 4}) || len(base) != 3 {
		t.Fatal("apply mutated the base set")
	}
	// A gap is corruption: slots are appended contiguously.
	if _, err := ApplyCentroidDeltas(base, []CentroidUpsert{{ID: 5, Vec: []float32{0, 0}}}); !errors.Is(err, ErrCorrupt) {
		t.Fatalf("gap: %v, want ErrCorrupt", err)
	}
}

func TestCentroidDeltaRejectsBadBytes(t *testing.T) {
	enc := EncodeCentroidDelta([]CentroidUpsert{{ID: 0, Vec: []float32{1, 2, 3}}})
	for name, c := range map[string]struct {
		data []byte
		dims int
	}{
		"wrong dims": {enc, 2},
		"truncated":  {enc[:len(enc)-2], 3},
		"trailing":   {append(append([]byte(nil), enc...), 0), 3},
		"magic":      {append([]byte("XXXX"), enc[4:]...), 3},
		"zero dims":  {enc, 0},
		"count lies": {append(enc[:8], 0xff, 0xff, 0xff, 0x7f), 3},
	} {
		if _, err := DecodeCentroidDelta(c.data, c.dims); !errors.Is(err, ErrCorrupt) {
			t.Fatalf("%s: %v, want ErrCorrupt", name, err)
		}
	}
}

// TestCentroidDeltaBounds: a split (one slot replaced, one appended) is a
// delta; an unchanged set is an empty one; a dropped slot or more than
// limit changes is a full set.
func TestCentroidDeltaBounds(t *testing.T) {
	prev := make([][]float32, 40)
	for i := range prev {
		prev[i] = []float32{float32(i), 1}
	}
	limit := max(2, len(prev)/16)
	next := append(slices.Clone(prev), []float32{100, 1})
	next[5] = []float32{5, 5}
	d, ok := CentroidDelta(prev, next, limit)
	if want := []CentroidUpsert{{ID: 5, Vec: next[5]}, {ID: 40, Vec: next[40]}}; !ok || !reflect.DeepEqual(d, want) {
		t.Fatalf("split delta: %v %v, want %v", d, ok, want)
	}
	if got, err := ApplyCentroidDeltas(prev, d); err != nil || !reflect.DeepEqual(got, next) {
		t.Fatalf("applied split delta: %v", err)
	}
	if d, ok := CentroidDelta(prev, prev, limit); !ok || len(d) != 0 {
		t.Fatalf("unchanged set: %v %v", d, ok)
	}
	if _, ok := CentroidDelta(prev, prev[1:], limit); ok {
		t.Fatal("a dropped slot must be a full set")
	}
	if _, ok := CentroidDelta(nil, prev, limit); ok {
		t.Fatal("an empty previous set must be a full set")
	}
	wide := slices.Clone(next)
	wide[7] = []float32{7, 7}
	if _, ok := CentroidDelta(prev, wide, limit); ok {
		t.Fatal("three changes past a limit of two must be a full set")
	}
}

// splitSet is a set of k well-separated centroids and the set a split of
// slot 5 leaves: slot 5 replaced by one half, the other appended.
func splitSet(k, dims int) (published, split [][]float32) {
	rng := rand.New(rand.NewSource(9))
	published = make([][]float32, k)
	for i := range published {
		published[i] = make([]float32, dims)
		for d := range published[i] {
			published[i][d] = float32(rng.NormFloat64() * 10)
		}
	}
	split = slices.Clone(published)
	a, b := slices.Clone(published[5]), slices.Clone(published[5])
	a[0] -= 1
	b[0] += 1
	split[5] = a
	return published, append(split, b)
}

const splitTestFanout = 100

// treeAssigner routes through tree at beam 16, falling back to the flat
// scan when the beam finds nothing.
func treeAssigner(tree *Tree, centroids [][]float32) func([]float32) int {
	s := tree.NewSearcher()
	return func(v []float32) int {
		if id := s.Assign(v, 16); id >= 0 {
			return id
		}
		return kmeans.Nearest(centroids, v)
	}
}

// TestUpsertAllAfterSplit: a split's tree is the old tree with the two
// changed slots upserted, not a rebuild, and it routes like a fresh Build
// of the split set.
func TestUpsertAllAfterSplit(t *testing.T) {
	const k, dims = 2000, 16
	published, split := splitSet(k, dims)
	tree, err := Build(context.Background(), published, splitTestFanout)
	if err != nil {
		t.Fatal(err)
	}
	limit := max(2, tree.Leaves()/16)
	up, ok := tree.UpsertAll(split, limit)
	if !ok || up == &tree || up.Leaves() != k+1 {
		t.Fatalf("UpsertAll: ok %v, %d leaves", ok, up.Leaves())
	}
	if tree.Leaves() != k || !slices.Equal(tree.Leaf(5), published[5]) {
		t.Fatal("the original tree was modified")
	}
	fresh, err := Build(context.Background(), split, splitTestFanout)
	if err != nil {
		t.Fatal(err)
	}
	upA, freshA := treeAssigner(up, split), treeAssigner(&fresh, split)
	rng := rand.New(rand.NewSource(10))
	differ := 0
	for n := range 5000 {
		base := split[rng.Intn(len(split))]
		if n%4 == 0 {
			base = split[5+(n/4%2)*(len(split)-6)] // the split's own halves
		}
		v := make([]float32, dims)
		for d := range v {
			v[d] = base[d] + float32(rng.NormFloat64()*0.5)
		}
		if upA(v) != freshA(v) {
			differ++
		}
	}
	if differ != 0 {
		t.Fatalf("%d of 5000 rows routed differently from a fresh Build", differ)
	}
	// The same split as a delta over the original tree lands on the same leaves.
	d, ok := CentroidDelta(published, split, limit)
	if !ok {
		t.Fatal("split is not a delta")
	}
	applied := tree.Clone()
	if err := applied.ApplyCentroidDeltas([][]CentroidUpsert{d}); err != nil {
		t.Fatal(err)
	}
	for i := range split {
		if !slices.Equal(applied.Leaf(i), up.Leaf(i)) {
			t.Fatalf("leaf %d differs between ApplyCentroidDeltas and UpsertAll", i)
		}
	}
	if err := applied.ApplyCentroidDeltas([][]CentroidUpsert{{{ID: k + 5, Vec: split[0]}}}); !errors.Is(err, ErrCorrupt) {
		t.Fatalf("gap into a tree: %v, want ErrCorrupt", err)
	}
	// A shorter set cuts the tree: the leaves are the set's.
	cut, ok := tree.UpsertAll(published[:k-1], limit)
	if !ok || cut.Leaves() != k-1 || tree.Leaves() != k {
		t.Fatalf("UpsertAll of a set one shorter: ok %v, %d leaves, original %d", ok, cut.Leaves(), tree.Leaves())
	}
	// Unchanged: the tree itself.
	if same, ok := tree.UpsertAll(published, limit); !ok || same != &tree {
		t.Fatal("an unchanged set did not reuse the tree")
	}
}

// BenchmarkSplitUpsertAll is the per-split cost of the tree at 10,150
// clusters: a full Build against clone + two upserts.
func BenchmarkSplitUpsertAll(b *testing.B) {
	const k, dims = 10_150, 128
	published, split := splitSet(k, dims)
	tree, err := Build(context.Background(), published, splitTestFanout)
	if err != nil {
		b.Fatal(err)
	}
	b.Run("build", func(b *testing.B) {
		for b.Loop() {
			if _, err := Build(context.Background(), split, splitTestFanout); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("upsert", func(b *testing.B) {
		for b.Loop() {
			if _, ok := tree.UpsertAll(split, max(2, k/16)); !ok {
				b.Fatal("not upserted")
			}
		}
	})
}

func TestCentroidAndMeanDistance(t *testing.T) {
	vecs := [][]float32{{0, 0}, {2, 0}, {1, 9, 9}} // the 3-wide row is skipped
	c, r := Centroid(vecs, 2, false)
	if !slices.Equal(c, []float32{1, 0}) || r != 1 {
		t.Fatalf("centroid %v radius %v, want [1 0] 1", c, r)
	}
	if c, r := Centroid([][]float32{{3, 4}}, 2, true); !slices.Equal(c, []float32{0.6, 0.8}) || r != 4 {
		t.Fatalf("spherical centroid %v radius %v, want [0.6 0.8] 4", c, r)
	}
	if c, r := Centroid(nil, 2, false); c != nil || r != 0 {
		t.Fatalf("empty: %v %v", c, r)
	}
	if r := MeanDistance([][]float32{{1}}, []float32{0, 0}); r != 0 {
		t.Fatalf("no matching rows: %v", r)
	}
}

// TestCentroidDeltaDropsBySwap: dropping a cluster by moving the last one
// into its slot is a two-entry delta (that upsert and a cut), not a full
// set; it round-trips, applies to the flat set and to the tree alike, and
// the cut tree routes like a fresh Build of the smaller set, never to the
// cut slot.
func TestCentroidDeltaDropsBySwap(t *testing.T) {
	const k, dims, drop = 2000, 16, 37
	published, _ := splitSet(k, dims)
	next := slices.Clone(published)
	next[drop] = next[k-1]
	next = next[:k-1]
	limit := max(2, k/16)
	d, ok := CentroidDelta(published, next, limit)
	if !ok || len(d) != 2 || d[0].ID != drop || d[1].ID != k-1 || d[1].Vec != nil {
		t.Fatalf("delta %v (ok %v): want the swap's upsert and a cut to %d", d, ok, k-1)
	}
	back, err := DecodeCentroidDelta(EncodeCentroidDelta(d), dims)
	if err != nil || len(back) != 2 || back[1].Vec != nil || back[1].ID != k-1 || !slices.Equal(back[0].Vec, d[0].Vec) {
		t.Fatalf("round trip: %v %v", back, err)
	}
	flat, err := ApplyCentroidDeltas(published, back)
	if err != nil || len(flat) != k-1 {
		t.Fatalf("flat apply: %d slots, %v", len(flat), err)
	}
	for i := range next {
		if !slices.Equal(flat[i], next[i]) {
			t.Fatalf("flat slot %d differs", i)
		}
	}
	tree, err := Build(context.Background(), published, splitTestFanout)
	if err != nil {
		t.Fatal(err)
	}
	applied := tree.Clone()
	if err := applied.ApplyCentroidDeltas([][]CentroidUpsert{back}); err != nil || applied.Leaves() != k-1 {
		t.Fatalf("tree apply: %d leaves, %v", applied.Leaves(), err)
	}
	for i := range next {
		if !slices.Equal(applied.Leaf(i), next[i]) {
			t.Fatalf("tree leaf %d differs", i)
		}
	}
	fresh, err := Build(context.Background(), next, splitTestFanout)
	if err != nil {
		t.Fatal(err)
	}
	cutA, freshA := treeAssigner(&applied, next), treeAssigner(&fresh, next)
	rng := rand.New(rand.NewSource(11))
	differ := 0
	for n := range 5000 {
		base := published[rng.Intn(k)]
		if n%4 == 0 {
			base = published[drop] // the dropped cluster's own region
		}
		v := make([]float32, dims)
		for j := range v {
			v[j] = base[j] + float32(rng.NormFloat64()*0.5)
		}
		got := cutA(v)
		if got < 0 || got >= k-1 {
			t.Fatalf("routed to slot %d of %d", got, k-1)
		}
		if got != freshA(v) {
			differ++
		}
	}
	if differ != 0 {
		t.Fatalf("%d of 5000 rows routed differently from a fresh Build", differ)
	}
	// One probe finds the moved centroid: the leaf moved holders with it.
	if got, _ := applied.Evaluations(next[drop], 1); len(got) != 1 || got[0] != drop {
		t.Fatalf("one probe for the moved centroid found %v, want [%d]", got, drop)
	}
	if err := applied.Truncate(k); err == nil {
		t.Fatal("a truncate past the leaves was accepted")
	}
}
