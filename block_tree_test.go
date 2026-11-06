package ivfq

import (
	"cmp"
	"context"
	"errors"
	"math"
	"math/rand"
	"slices"
	"testing"

	"github.com/x448/float16"
)

// A Dense[float32] tree routes exactly as FloatTree: same topology, same
// float32 distances, same tie order, at every beam and probe.
func TestTreeDenseFloat32MatchesFloatTree(t *testing.T) {
	ctx := context.Background()
	vs := treeRand(2000, 12, 3)
	vs = append(vs, vs[5], vs[5], vs[1999]) // duplicate leaves: exact ties
	ft, err := Build(ctx, vs, 6)
	if err != nil {
		t.Fatal(err)
	}
	if ft.Height() < 4 {
		t.Fatalf("height %d: want a deep tree", ft.Height())
	}
	tr, err := BuildTree(ctx, vs, 6, NewDense[float32])
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(1))
	qs := treeRand(100, 12, 9)
	for range 20 {
		qs = append(qs, vs[rng.Intn(len(vs))])
	}
	for _, q := range qs {
		for _, beam := range []int{1, 3, 8, 64} {
			for _, probe := range []int{1, 5, 40} {
				want, _ := ft.EvaluationsBeam(q, beam, probe)
				if got := tr.Nearest(q, beam, probe); !slices.Equal(got, want) {
					t.Fatalf("beam %d probe %d: got %v, FloatTree %v", beam, probe, got, want)
				}
			}
		}
	}
}

// With a beam covering every node, a tree over small integer coordinates,
// held exactly in float32 and float16, ranks every leaf in exact
// (distance, original id) order across many levels and blocks.
func TestTreeLeafIdentityAndTies(t *testing.T) {
	rng := rand.New(rand.NewSource(4))
	vs := make([][]float32, 300) // 64 distinct points: many exact ties
	for i := range vs {
		vs[i] = []float32{float32(rng.Intn(4)), float32(rng.Intn(4)), float32(rng.Intn(4))}
	}
	type leaf struct{ id, d int }
	exact := func(q []float32) []int {
		ls := make([]leaf, len(vs))
		for id, v := range vs {
			d := 0
			for j := range v {
				x := int(q[j] - v[j])
				d += x * x
			}
			ls[id] = leaf{id, d}
		}
		slices.SortFunc(ls, func(a, b leaf) int { return cmp.Or(cmp.Compare(a.d, b.d), cmp.Compare(a.id, b.id)) })
		ids := make([]int, len(ls))
		for i, l := range ls {
			ids[i] = l.id
		}
		return ids
	}
	check := func(name string, nearest func(q []float32, beam, probe int) []int) {
		for range 50 {
			q := []float32{float32(rng.Intn(5) - 1), float32(rng.Intn(5) - 1), float32(rng.Intn(5) - 1)}
			want := exact(q)
			if got := nearest(q, len(vs), len(vs)); !slices.Equal(got, want) {
				t.Fatalf("%s q %v: got %v, want %v", name, q, got, want)
			}
			if got := nearest(q, len(vs), 7); !slices.Equal(got, want[:7]) {
				t.Fatalf("%s q %v probe 7: got %v, want %v", name, q, got, want[:7])
			}
		}
	}
	ft := mustBuild(vs, 4)
	if ft.Height() < 3 {
		t.Fatalf("height %d: want several levels", ft.Height())
	}
	f32, err := BuildTree(context.Background(), vs, 4, NewDense[float32])
	if err != nil {
		t.Fatal(err)
	}
	f16, err := BuildTree(context.Background(), vs, 4, NewDense[float16.Float16])
	if err != nil {
		t.Fatal(err)
	}
	check("float32", f32.Nearest)
	check("float16", f16.Nearest)
}

// BuildTree keeps nothing of the caller's centroids, returns fresh results,
// and fails on bad input, a failing encoder, a block that disagrees with
// its rows, or a cancelled ctx.
func TestBuildTreeContract(t *testing.T) {
	ctx := context.Background()
	vs := treeRand(300, 8, 5)
	q := slices.Clone(vs[17])
	before := treeRand(300, 8, 5)
	tr, err := BuildTree(ctx, vs, 4, NewDense[float32])
	if err != nil {
		t.Fatal(err)
	}
	for i := range vs {
		if !slices.Equal(vs[i], before[i]) {
			t.Fatalf("BuildTree changed centroid %d", i)
		}
	}
	want := tr.Nearest(q, 4, 10)
	for _, v := range vs {
		clear(v)
	}
	got := tr.Nearest(q, 4, 10)
	if !slices.Equal(got, want) || tr.Leaves() != 300 || tr.Dims() != 8 {
		t.Fatalf("after clearing the input: %v, want %v", got, want)
	}
	got[0] = -1
	if again := tr.Nearest(q, 4, 10); !slices.Equal(again, want) {
		t.Fatalf("results alias: %v", again)
	}
	if tr.Nearest(q[:7], 4, 10) != nil || tr.Nearest(q, 0, 10) != nil || tr.Nearest(q, 4, 0) != nil {
		t.Fatal("bad width, beam or probe returned leaves")
	}

	vs = treeRand(300, 8, 5)
	boom := errors.New("boom")
	calls := 0
	failing := func(rows [][]float32) (Dense[float32], error) {
		if calls++; calls == 5 {
			return Dense[float32]{}, boom
		}
		return NewDense[float32](rows)
	}
	if _, err := BuildTree(ctx, vs, 4, failing); !errors.Is(err, boom) {
		t.Fatalf("encoder error: %v", err)
	}
	short := func(rows [][]float32) (Dense[float32], error) { return NewDense[float32](rows[:1]) }
	if _, err := BuildTree(ctx, vs, 4, short); err == nil {
		t.Fatal("a block with too few rows was accepted")
	}
	narrow := func(rows [][]float32) (Dense[float32], error) {
		cut := make([][]float32, len(rows))
		for i, r := range rows {
			cut[i] = r[:7]
		}
		return NewDense[float32](cut)
	}
	if _, err := BuildTree(ctx, vs, 4, narrow); err == nil {
		t.Fatal("a block of the wrong width was accepted")
	}
	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	if _, err := BuildTree(cancelled, vs, 4, NewDense[float32]); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled build: %v", err)
	}
	if _, err := BuildTree(cancelled, nil, 4, NewDense[float32]); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled empty build: %v", err)
	}
	last, cancelLast := context.WithCancel(ctx)
	cancelInEncode := func(rows [][]float32) (Dense[float32], error) { cancelLast(); return NewDense[float32](rows) }
	if _, err := BuildTree(last, vs[:1], 4, cancelInEncode); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled during the only encode: %v", err)
	}
	if _, err := BuildTree(ctx, [][]float32{{}, {}}, 4, NewDense[float32]); err == nil {
		t.Fatal("zero-dimensional centroids were accepted")
	}
	vs[3][2] = float32(math.NaN())
	if _, err := BuildTree(ctx, vs, 4, NewDense[float32]); err == nil {
		t.Fatal("a NaN centroid was accepted")
	}
	empty, err := BuildTree(ctx, nil, 4, NewDense[float32])
	if err != nil || empty.Nearest(nil, 1, 1) != nil || empty.Leaves() != 0 {
		t.Fatalf("empty tree: %v", err)
	}
}
