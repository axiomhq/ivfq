package rabitq

import (
	"bytes"
	"context"
	"math"
	"math/rand"
	"slices"
	"sort"
	"sync"
	"testing"

	"github.com/axiomhq/ivfq"
	"github.com/axiomhq/ivfq/vector"
	"github.com/x448/float16"
)

// blockCorpus is n leaf centroids around 32 shared centers, with unit noise.
func blockCorpus(n, dims int, seed int64) [][]float32 {
	rng := rand.New(rand.NewSource(seed))
	centers := make([][]float32, 32)
	for i := range centers {
		centers[i] = make([]float32, dims)
		for j := range centers[i] {
			centers[i][j] = float32(3 * rng.NormFloat64())
		}
	}
	out := make([][]float32, n)
	for i := range out {
		c := centers[rng.Intn(len(centers))]
		out[i] = make([]float32, dims)
		for j := range out[i] {
			out[i][j] = c[j] + float32(rng.NormFloat64())
		}
	}
	return out
}

// A Tree of RaBitQ blocks routes by estimates, so it is held to recall,
// not equality: the exact top 10 by squared distance over the raw leaf
// centroids, computed here in float64, must mostly fall within the tree's
// 40 nearest. The same tree answers identically from concurrent searches.
func TestBlockTreeRecallAgainstRawCentroids(t *testing.T) {
	const dims, topk, probe = 64, 10, 40
	ctx := context.Background()
	leaves := blockCorpus(1024, dims, 1)
	queries := blockCorpus(100, dims, 2)
	rot := NewRotation(Seed("block", "test"), dims)
	tree, err := ivfq.BuildTree(ctx, leaves, 16, func(rows [][]float32) (Block, error) { return NewBlock(rows, rot) })
	if err != nil {
		t.Fatal(err)
	}
	covered := 0
	got := make([][]int, len(queries))
	for i, q := range queries {
		ids := make([]int, len(leaves))
		d := make([]float64, len(leaves))
		for id, v := range leaves {
			ids[id] = id
			for j := range v {
				x := float64(q[j]) - float64(v[j])
				d[id] += x * x
			}
		}
		sort.SliceStable(ids, func(a, b int) bool { return d[ids[a]] < d[ids[b]] })
		got[i] = tree.Nearest(q, 64, probe)
		for _, id := range ids[:topk] {
			if slices.Contains(got[i], id) {
				covered++
			}
		}
	}
	// Measured 0.97 here (0.94 to 0.999 across synthetic and Big ANN
	// centroid sets); a sign, scale or id mistake scores near zero.
	if r := float64(covered) / float64(topk*len(queries)); r < 0.9 {
		t.Fatalf("exact top %d within the tree's %d: recall %.3f", topk, probe, r)
	}

	half, err := ivfq.BuildTree(ctx, leaves, 16, ivfq.NewDense[float16.Float16])
	if err != nil {
		t.Fatal(err)
	}
	halfWant := make([][]int, len(queries))
	for i, q := range queries {
		halfWant[i] = half.Nearest(q, 8, topk)
	}
	var wg sync.WaitGroup
	for range 8 {
		wg.Go(func() {
			for i, q := range queries {
				if !slices.Equal(tree.Nearest(q, 64, probe), got[i]) || !slices.Equal(half.Nearest(q, 8, topk), halfWant[i]) {
					t.Error("a concurrent search disagreed")
					return
				}
			}
		})
	}
	wg.Wait()
}

func TestBlockShape(t *testing.T) {
	rot := NewRotation(3, 4)
	rows := [][]float32{{1, 2, 3, 4}, {-1, 0, 2, 5}, {0, 0, 0, 0}}
	b, err := NewBlock(rows, rot)
	if err != nil {
		t.Fatal(err)
	}
	if b.Len() != 3 || b.Dims() != 4 {
		t.Fatalf("shape %d x %d", b.Len(), b.Dims())
	}
	q := []float32{0.5, -1, 2, 3}
	scores := []float32{7, 7, 7, 7, 7}
	b.ScoreAll(q, scores)
	if scores[3] != 7 || scores[4] != 7 || !slices.Equal(q, []float32{0.5, -1, 2, 3}) {
		t.Fatalf("ScoreAll wrote past Len or into the query: %v %v", scores, q)
	}
	if _, err := NewBlock([][]float32{{1, 2, 3}}, rot); err == nil {
		t.Fatal("a row of the wrong width was accepted")
	}
	if _, err := NewBlock(rows, nil); err == nil {
		t.Fatal("a nil rotation was accepted")
	}
	if _, err := NewBlock([][]float32{{1, 2, 3, float32(math.NaN())}}, rot); err == nil {
		t.Fatal("a NaN row was accepted")
	}
}

var _ ivfq.Encoder[uint8, Block] = BlockEncoder[uint8]{}

// blockSource is rows x dims random values of S, signed and fractional
// where S allows, uint8 spanning past 127.
func blockSource[S vector.Scalar](rows, dims int, seed int64) vector.Slice[S] {
	rng := rand.New(rand.NewSource(seed))
	data := make([]S, rows*dims)
	for i := range data {
		x := rng.NormFloat64() * 40
		switch p := any(&data[i]).(type) {
		case *float32:
			*p = float32(x)
		case *float16.Float16:
			*p = float16.Fromfloat32(float32(x))
		case *int8:
			*p = int8(max(-128, min(127, math.Round(x))))
		case *uint8:
			*p = uint8(max(0, min(255, math.Round(x+128))))
		}
	}
	return vector.Slice[S]{Dims: dims, Data: data}
}

// BlockEncoder makes exactly NewBlock's codes from any source scalar,
// over more rows than one fill batch, without changing or keeping its
// input.
func TestBlockEncoder(t *testing.T) {
	checkBlockEncoder(t, blockSource[float32](70, 40, 1))
	checkBlockEncoder(t, blockSource[float16.Float16](70, 40, 2))
	checkBlockEncoder(t, blockSource[int8](70, 40, 3))
	checkBlockEncoder(t, blockSource[uint8](70, 40, 4))
	checkBlockEncoder(t, vector.Slice[int8]{Dims: 40}) // no rows, as NewBlock allows

	rot := NewRotation(5, 40)
	ok := blockSource[float32](3, 40, 5)
	nan := blockSource[float32](3, 40, 5)
	nan.Data[77] = float32(math.NaN())
	inf := blockSource[float16.Float16](3, 40, 6)
	inf.Data[5] = float16.Frombits(0xFC00)
	for name, err := range map[string]error{
		"nil rotation":        encodeBlockErr(BlockEncoder[float32]{}, ok),
		"rotation too narrow": encodeBlockErr(BlockEncoder[float32]{NewRotation(5, 39)}, ok),
		"zero dims":           encodeBlockErr(BlockEncoder[int8]{rot}, vector.Slice[int8]{Data: []int8{1}}),
		"partial row":         encodeBlockErr(BlockEncoder[uint8]{rot}, vector.Slice[uint8]{Dims: 40, Data: make([]uint8, 41)}),
		"float32 NaN":         encodeBlockErr(BlockEncoder[float32]{rot}, nan),
		"float16 -Inf":        encodeBlockErr(BlockEncoder[float16.Float16]{rot}, inf),
	} {
		if err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}

func encodeBlockErr[S vector.Scalar](e BlockEncoder[S], s vector.Slice[S]) error {
	_, err := e.Encode(s)
	return err
}

func checkBlockEncoder[S vector.Scalar](t *testing.T, src vector.Slice[S]) {
	t.Helper()
	rot := NewRotation(Seed("encoder"), src.Dims)
	before := slices.Clone(src.Data)
	got, err := BlockEncoder[S]{Rotation: rot}.Encode(src)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(src.Data, before) {
		t.Fatal("Encode changed its input")
	}
	rows, err := src.Float32Rows()
	if err != nil {
		t.Fatal(err)
	}
	want, err := NewBlock(rows, rot)
	if err != nil {
		t.Fatal(err)
	}
	gotBytes, _ := got.col.MarshalBinary()
	clear(src.Data)
	wantBytes, _ := want.col.MarshalBinary()
	if !bytes.Equal(gotBytes, wantBytes) || got.Len() != want.Len() || got.Dims() != want.Dims() {
		t.Fatalf("%T: Encode's codes differ from NewBlock's", src.Data)
	}
	if after, _ := got.col.MarshalBinary(); !bytes.Equal(after, gotBytes) {
		t.Fatalf("%T: the block changed with its input", src.Data)
	}
	q := blockCorpus(1, src.Dims, 9)[0]
	a, b := make([]float32, got.Len()), make([]float32, got.Len())
	got.ScoreAll(q, a)
	want.ScoreAll(q, b)
	if !slices.EqualFunc(a, b, func(x, y float32) bool { return math.Float32bits(x) == math.Float32bits(y) }) {
		t.Fatalf("%T: scores differ from NewBlock's", src.Data)
	}
}
