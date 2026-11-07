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

// dq is q as a dense float32 query.
func dq(t testing.TB, q []float32) ivfq.Embedding {
	t.Helper()
	d, err := ivfq.NewDense[float32]([][]float32{q})
	if err != nil {
		t.Fatal(err)
	}
	return d.At(0)
}

// nearest is tr.Nearest(dq(q), beam, probe), failing t on an error.
func nearest[B ivfq.EmbeddingBatch](t testing.TB, tr ivfq.Tree[B], q []float32, beam, probe int) []int {
	t.Helper()
	ids, err := tr.Nearest(dq(t, q), beam, probe)
	if err != nil {
		t.Error(err)
	}
	return ids
}

// scoreL2 is b.ScoreAll(dq(q), L2, dst), failing t on an error.
func scoreL2(t testing.TB, b ivfq.EmbeddingBatch, q []float32, dst []float32) {
	t.Helper()
	if err := b.ScoreAll(dq(t, q), ivfq.L2, dst); err != nil {
		t.Fatal(err)
	}
}

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
		got[i] = nearest(t, tree, q, 64, probe)
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
		halfWant[i] = nearest(t, half, q, 8, topk)
	}
	var wg sync.WaitGroup
	for range 8 {
		wg.Go(func() {
			for i, q := range queries {
				if !slices.Equal(nearest(t, tree, q, 64, probe), got[i]) || !slices.Equal(nearest(t, half, q, 8, topk), halfWant[i]) {
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
	scoreL2(t, b, q, scores)
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
	scoreL2(t, got, q, a)
	scoreL2(t, want, q, b)
	if !slices.EqualFunc(a, b, func(x, y float32) bool { return math.Float32bits(x) == math.Float32bits(y) }) {
		t.Fatalf("%T: scores differ from NewBlock's", src.Data)
	}
}

// typedQuery is q as a dense query of scalar T, and the float32 values it
// decodes to.
func typedQuery[T vector.Scalar](t *testing.T, q []float32) (ivfq.Embedding, []float32) {
	t.Helper()
	d, err := ivfq.NewDense[T]([][]float32{q})
	if err != nil {
		t.Fatal(err)
	}
	e := d.At(0).(ivfq.DenseEmbedding[T])
	v := make([]float32, len(q))
	if err := e.Decode(v); err != nil {
		t.Fatal(err)
	}
	return e, v
}

// A Block scores every dense query type as the RaBitQ L2 estimate of the
// values the query decodes to, the Scorer's estimate bit for bit (not its
// bound); At(i).Score agrees; ScoreAll writes only dst[:Len()].
func TestBlockScoresTypedQueries(t *testing.T) {
	const dims = 48
	rot := NewRotation(Seed("typed", "query"), dims)
	rows := blockCorpus(40, dims, 11)
	b, err := NewBlock(rows, rot)
	if err != nil {
		t.Fatal(err)
	}
	raw := blockCorpus(1, dims, 12)[0]
	for j := range raw {
		raw[j] *= 10 // past 127 and below zero once rounded to integers
	}
	var qs []ivfq.Embedding
	var vals [][]float32
	for _, f := range []func(*testing.T, []float32) (ivfq.Embedding, []float32){
		typedQuery[float32], typedQuery[float16.Float16], typedQuery[int8], typedQuery[uint8],
	} {
		q, v := f(t, raw)
		qs, vals = append(qs, q), append(vals, v)
	}
	for k, q := range qs {
		want := make([]float32, b.Len())
		sc := b.col.Scorer(NewQuery(vals[k], ivfq.L2, rot))
		for i := range want {
			want[i] = sc.Score(i)
		}
		dst := make([]float32, b.Len()+2)
		dst[b.Len()], dst[b.Len()+1] = 9, 9
		if err := b.ScoreAll(q, ivfq.L2, dst); err != nil {
			t.Fatalf("%T: %v", q, err)
		}
		if dst[b.Len()] != 9 || dst[b.Len()+1] != 9 {
			t.Fatalf("%T: ScoreAll wrote past Len", q)
		}
		for i := range want {
			one, err := b.At(i).Score(q, ivfq.L2)
			if err != nil || math.Float32bits(dst[i]) != math.Float32bits(want[i]) || one != dst[i] {
				t.Fatalf("%T row %d: ScoreAll %v, Score %v (%v), Scorer %v", q, i, dst[i], one, err, want[i])
			}
		}
	}
}

// Dense and RaBitQ scores of one query over the same rows are on one
// scale: the estimate tracks the exact -‖q-row‖², sign and magnitude, so a
// caller may compare scores across the two encodings.
func TestBlockAndDenseScoresComparable(t *testing.T) {
	const dims = 96
	rows := blockCorpus(300, dims, 21)
	b, err := NewBlock(rows, NewRotation(Seed("mixed"), dims))
	if err != nil {
		t.Fatal(err)
	}
	d, err := ivfq.NewDense[float32](rows)
	if err != nil {
		t.Fatal(err)
	}
	var off, mag float64
	for _, q := range blockCorpus(20, dims, 22) {
		est, exact := make([]float32, len(rows)), make([]float32, len(rows))
		scoreL2(t, b, q, est)
		scoreL2(t, d, q, exact)
		for i, r := range rows {
			var want float64
			for j := range r {
				x := float64(q[j]) - float64(r[j])
				want -= x * x
			}
			if math.Abs(float64(exact[i])-want) > 1e-4*math.Abs(want) {
				t.Fatalf("Dense row %d: %v, want %v", i, exact[i], want)
			}
			off += math.Abs(float64(est[i]) - want)
			mag += math.Abs(want)
		}
	}
	// Measured 0.062; a sign, scale or frame mistake is near 1 or more.
	if r := off / mag; r > 0.15 {
		t.Fatalf("RaBitQ estimates are off by %.3f of the exact scores", r)
	}
}

// A Block fails, rather than scoring, for a metric its L2 codes do not
// hold, a query of another width or representation, a short dst or a zero
// Block; At panics outside the rows.
func TestBlockScoreErrors(t *testing.T) {
	rot := NewRotation(3, 4)
	b, err := NewBlock([][]float32{{1, 2, 3, 4}, {-1, 0, 2, 5}}, rot)
	if err != nil {
		t.Fatal(err)
	}
	q := dq(t, []float32{0.5, -1, 2, 3})
	ok := make([]float32, 2)
	for name, err := range map[string]error{
		"cosine":        b.ScoreAll(q, ivfq.Cosine, ok),
		"inner product": b.ScoreAll(q, ivfq.InnerProduct, ok),
		"3-wide query":  b.ScoreAll(dq(t, []float32{1, 2, 3}), ivfq.L2, ok),
		"code query":    b.ScoreAll(b.At(0), ivfq.L2, ok),
		"nil query":     b.ScoreAll(nil, ivfq.L2, ok),
		"typed nil":     b.ScoreAll((*ivfq.DenseEmbedding[float32])(nil), ivfq.L2, ok),
		"short dst":     b.ScoreAll(q, ivfq.L2, ok[:1]),
		"zero Block":    Block{}.ScoreAll(q, ivfq.L2, ok),
		"row cosine":    func() error { _, err := b.At(1).Score(q, ivfq.Cosine); return err }(),
		"row as query":  func() error { _, err := b.At(1).Score(b.At(0), ivfq.L2); return err }(),
	} {
		if err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
	for _, i := range []int{-1, 2} {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("At(%d) of 2 rows did not panic", i)
				}
			}()
			b.At(i)
		}()
	}
}

// BenchmarkBlockScoreAll is one dense query against a 256 x 128 Block,
// the per-node work of a RaBitQ tree search, query decode and preparation
// included.
func BenchmarkBlockScoreAll(b *testing.B) {
	rows := blockCorpus(256, 128, 1)
	blk, err := NewBlock(rows, NewRotation(Seed("bench"), 128))
	if err != nil {
		b.Fatal(err)
	}
	dst := make([]float32, len(rows))
	for _, name := range []string{"float32", "int8"} {
		var q ivfq.Embedding
		raw := blockCorpus(1, 128, 2)[0]
		if name == "float32" {
			q = dq(b, raw)
		} else {
			d, err := ivfq.NewDense[int8]([][]float32{raw})
			if err != nil {
				b.Fatal(err)
			}
			q = d.At(0)
		}
		b.Run(name, func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				if err := blk.ScoreAll(q, ivfq.L2, dst); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
