package ivfq

import (
	"math"
	"math/rand"
	"slices"
	"sync"
	"testing"

	"github.com/axiomhq/ivfq/vector"
	"github.com/x448/float16"
)

// dq is q as a dense float32 query.
func dq(t testing.TB, q []float32) Embedding {
	t.Helper()
	d, err := NewDense[float32]([][]float32{q})
	if err != nil {
		t.Fatal(err)
	}
	return d.At(0)
}

// scoreL2 is b.ScoreAll(dq(q), L2, dst), failing t on an error.
func scoreL2(t testing.TB, b EmbeddingBatch, q []float32, dst []float32) {
	t.Helper()
	if err := b.ScoreAll(dq(t, q), L2, dst); err != nil {
		t.Fatal(err)
	}
}

// treeNearest is tr.Nearest(dq(q), beam, probe), failing t on an error.
func treeNearest[B EmbeddingBatch](t testing.TB, tr Tree[B], q []float32, beam, probe int) []int {
	t.Helper()
	ids, err := tr.Nearest(dq(t, q), beam, probe)
	if err != nil {
		t.Fatal(err)
	}
	return ids
}

// denseScores encodes rows as T and scores q against them, checking that
// ScoreAll writes only Len() scores and leaves q alone.
func denseScores[T vector.Scalar](t *testing.T, rows [][]float32, q []float32) []float32 {
	t.Helper()
	d, err := NewDense[T](rows)
	if err != nil {
		t.Fatal(err)
	}
	if d.Len() != len(rows) || d.Dims() != len(rows[0]) {
		t.Fatalf("shape %d x %d", d.Len(), d.Dims())
	}
	scores := []float32{7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7}[:len(rows)+2]
	q0 := slices.Clone(q)
	scoreL2(t, d, q, scores)
	if scores[len(rows)] != 7 || scores[len(rows)+1] != 7 || !slices.Equal(q, q0) {
		t.Fatalf("ScoreAll wrote past Len or into the query: %v %v", scores, q)
	}
	return scores[:len(rows)]
}

// wantScores is -‖q - row‖² in float64 over the values a block should hold.
func wantScores(q []float32, rows [][]float64) []float64 {
	out := make([]float64, len(rows))
	for i, r := range rows {
		for j, x := range r {
			out[i] -= (float64(q[j]) - x) * (float64(q[j]) - x)
		}
	}
	return out
}

func checkScores(t *testing.T, name string, got []float32, want []float64, q []float32, rows [][]float64, slack []float64) {
	t.Helper()
	for i := range want {
		mag := 0.0
		for j, x := range rows[i] {
			mag += float64(q[j])*float64(q[j]) + x*x
		}
		if d := math.Abs(float64(got[i]) - want[i]); !(d <= 1e-6*mag+1e-6+slack[i]) { // NaN fails
			t.Fatalf("%s row %d: score %v, want %v (off %v)", name, i, got[i], want[i], d)
		}
	}
}

// Each T holds the value its documented conversion gives: float16 rounds
// to the nearest half (and is not its bits read as a number), int8 is
// signed, uint8 holds values above 127, and integer codes round halves
// away from zero on a block scale that is 1 when the block spans T's range.
func TestDenseConversions(t *testing.T) {
	q := []float32{0.25, -1, 3, 7, -300, 2, 0.5, -0.75}
	zero := make([]float32, 8)

	f16 := [][]float32{{1, -2, 0.1, 65504, 1.0 / 3, 1e-8, 2048.5, -0.5}}
	f16Want := [][]float64{{1, -2, 0.0999755859375, 65504, 0.333251953125, 0, 2048, -0.5}}
	i8 := [][]float32{{-128, 127, -100, 3.4, 3.6, 2.5, -2.5, 0}, {0.49, -0.5, 99.5, -1, 1, 64, -64, 126.6}}
	i8Want := [][]float64{{-128, 127, -100, 3, 4, 3, -3, 0}, {0, -1, 100, -1, 1, 64, -64, 127}}
	u8 := [][]float32{{0, 255, 200, 127.5, 0.49, 1.5, 254.6, 128}, {17, 16.5, 99.4, 129, 3, 250, 0, 1}}
	u8Want := [][]float64{{0, 255, 200, 128, 0, 2, 255, 128}, {17, 17, 99, 129, 3, 250, 0, 1}}
	none := []float64{0, 0}
	for _, qq := range [][]float32{q, zero} {
		checkScores(t, "float16", denseScores[float16.Float16](t, f16, qq), wantScores(qq, f16Want), qq, f16Want, none)
		checkScores(t, "int8", denseScores[int8](t, i8, qq), wantScores(qq, i8Want), qq, i8Want, none)
		checkScores(t, "uint8", denseScores[uint8](t, u8, qq), wantScores(qq, u8Want), qq, u8Want, none)
		exact := [][]float64{{1, -2, 0.1, 65504, 1.0 / 3, 1e-8, 2048.5, -0.5}}
		for i, x := range f16[0] {
			exact[0][i] = float64(x)
		}
		checkScores(t, "float32", denseScores[float32](t, f16, qq), wantScores(qq, exact), qq, exact, none)
	}

	// A block of equal coordinates decodes exactly, at any value.
	flat := [][]float32{{5.25, 5.25, 5.25, 5.25, 5.25, 5.25, 5.25, 5.25}, {5.25, 5.25, 5.25, 5.25, 5.25, 5.25, 5.25, 5.25}}
	flatWant := [][]float64{{5.25, 5.25, 5.25, 5.25, 5.25, 5.25, 5.25, 5.25}, {5.25, 5.25, 5.25, 5.25, 5.25, 5.25, 5.25, 5.25}}
	checkScores(t, "int8 flat", denseScores[int8](t, flat, q), wantScores(q, flatWant), q, flatWant, none)
	checkScores(t, "uint8 flat", denseScores[uint8](t, flat, q), wantScores(q, flatWant), q, flatWant, none)

	// A narrow fractional block is held to half a step of (max-min)/255,
	// not rounded to whole numbers.
	rng := rand.New(rand.NewSource(2))
	narrow := make([][]float32, 6)
	orig := make([][]float64, len(narrow))
	lo, hi := math.Inf(1), math.Inf(-1)
	for i := range narrow {
		narrow[i] = make([]float32, 8)
		orig[i] = make([]float64, 8)
		for j := range narrow[i] {
			x := float32(-10 + 0.5*rng.Float64())
			narrow[i][j], orig[i][j] = x, float64(x)
			lo, hi = min(lo, float64(x)), max(hi, float64(x))
		}
	}
	want := wantScores(q, orig)
	e := math.Sqrt(8) * (hi - lo) / 255 / 2 * 1.001
	slack := make([]float64, len(want))
	for i := range slack {
		slack[i] = 2*math.Sqrt(-want[i])*e + e*e
	}
	checkScores(t, "int8 narrow", denseScores[int8](t, narrow, q), want, q, orig, slack)
	checkScores(t, "uint8 narrow", denseScores[uint8](t, narrow, q), want, q, orig, slack)

	for name, rows := range map[string][][]float32{
		"NaN":             {{1, float32(math.NaN())}},
		"Inf":             {{float32(math.Inf(-1)), 1}},
		"ragged":          {{1, 2}, {1}},
		"empty":           {},
		"0-dim":           {{}},
		"MaxFloat32 norm": {{math.MaxFloat32}},
		"2e19 norm":       {{2e19, 0}},
		"int8 decoded":    {{-3e38, 3e38}, {1, -1}},
	} {
		if name == "int8 decoded" {
			if _, err := NewDense[int8](rows); err == nil {
				t.Fatal("an int8 block whose decoded norms overflow was accepted")
			}
			continue
		}
		_, e1 := NewDense[float32](rows)
		_, e2 := NewDense[float16.Float16](rows)
		_, e3 := NewDense[int8](rows)
		_, e4 := NewDense[uint8](rows)
		if e1 == nil || e2 == nil || e3 == nil || e4 == nil {
			t.Fatalf("%s rows accepted: %v %v %v %v", name, e1, e2, e3, e4)
		}
	}
	if _, err := NewDense[float16.Float16]([][]float32{{65520}}); err == nil {
		t.Fatal("65520 rounds past float16's range and was accepted")
	}
}

var _ Encoder[int8, Dense[float16.Float16]] = DenseEncoder[int8, float16.Float16]{}

// DenseEncoder reads every source scalar by value (float16 as a half, int8
// signed, uint8 past 127), keeps nothing of its input, and encodes exactly
// as NewDense over the source's Float32Rows does, for every output type.
func TestDenseEncoder(t *testing.T) {
	half := func(bits ...uint16) []float16.Float16 {
		out := make([]float16.Float16, len(bits))
		for i, b := range bits {
			out[i] = float16.Frombits(b)
		}
		return out
	}
	checkDenseEncoder(t, vector.Slice[float32]{Dims: 4, Data: []float32{0.1, -2.5, 1000, 3.75, -7, 0, 0.5, 2e-3}},
		[][]float64{{float64(float32(0.1)), -2.5, 1000, 3.75}, {-7, 0, 0.5, float64(float32(2e-3))}})
	// As integers these bits are 13653, 49408, 25600, 1, 48128, 0, 14336, 31743.
	checkDenseEncoder(t, vector.Slice[float16.Float16]{Dims: 4, Data: half(0x3555, 0xC100, 0x6400, 0x0001, 0xBC00, 0x0000, 0x3800, 0x7BFF)},
		[][]float64{{0.333251953125, -2.5, 1024, math.Ldexp(1, -24)}, {-1, 0, 0.5, 65504}})
	checkDenseEncoder(t, vector.Slice[int8]{Dims: 4, Data: []int8{-128, 127, -3, 5, 0, -1, 100, -100}},
		[][]float64{{-128, 127, -3, 5}, {0, -1, 100, -100}})
	checkDenseEncoder(t, vector.Slice[uint8]{Dims: 4, Data: []uint8{0, 255, 128, 200, 1, 127, 129, 64}},
		[][]float64{{0, 255, 128, 200}, {1, 127, 129, 64}})

	for name, errs := range map[string][4]error{
		"zero dims":         encodeErr(vector.Slice[int8]{Data: []int8{1}}),
		"negative dims":     encodeErr(vector.Slice[uint8]{Dims: -1, Data: []uint8{1}}),
		"partial row":       encodeErr(vector.Slice[int8]{Dims: 2, Data: []int8{1, 2, 3}}),
		"no rows":           encodeErr(vector.Slice[uint8]{Dims: 3}),
		"float32 NaN":       encodeErr(vector.Slice[float32]{Dims: 2, Data: []float32{1, float32(math.NaN())}}),
		"float32 -Inf":      encodeErr(vector.Slice[float32]{Dims: 1, Data: []float32{float32(math.Inf(-1))}}),
		"float16 NaN":       encodeErr(vector.Slice[float16.Float16]{Dims: 2, Data: half(0x3C00, 0x7E00)}),
		"float16 +Inf":      encodeErr(vector.Slice[float16.Float16]{Dims: 1, Data: half(0x7C00)}),
		"norm past float32": encodeErr(vector.Slice[float32]{Dims: 2, Data: []float32{2e19, 0}}),
	} {
		for i, err := range errs {
			if err == nil {
				t.Errorf("%s: accepted as Dense[%s]", name, denseOutputs[i])
			}
		}
	}
	if _, err := (DenseEncoder[float32, float16.Float16]{}).Encode(vector.Slice[float32]{Dims: 1, Data: []float32{70000}}); err == nil {
		t.Error("70000 encoded as a float16 block")
	}
}

var denseOutputs = [4]string{"float32", "float16", "int8", "uint8"}

// encodeErr is the error of encoding s into each Dense type, in
// denseOutputs order, so a caller checks every output type on its own.
func encodeErr[S vector.Scalar](s vector.Slice[S]) [4]error {
	_, e1 := DenseEncoder[S, float32]{}.Encode(s)
	_, e2 := DenseEncoder[S, float16.Float16]{}.Encode(s)
	_, e3 := DenseEncoder[S, int8]{}.Encode(s)
	_, e4 := DenseEncoder[S, uint8]{}.Encode(s)
	return [4]error{e1, e2, e3, e4}
}

func checkDenseEncoder[S vector.Scalar](t *testing.T, src vector.Slice[S], values [][]float64) {
	t.Helper()
	qs := [][]float32{{0.25, -1, 3, 7}, {-300, 2, 0.5, 60000}, {0, 0, 0, 0}}
	exact, err := DenseEncoder[S, float32]{}.Encode(src)
	if err != nil {
		t.Fatal(err)
	}
	for _, q := range qs {
		got := make([]float32, exact.Len())
		scoreL2(t, exact, q, got)
		checkScores(t, "Dense[float32] of the source", got, wantScores(q, values), q, values, make([]float64, len(values)))
	}
	sameAsNewDense[S, float32](t, src, qs)
	sameAsNewDense[S, float16.Float16](t, src, qs)
	sameAsNewDense[S, int8](t, src, qs)
	sameAsNewDense[S, uint8](t, src, qs)
}

// sameAsNewDense checks that Encode scores bit for bit as NewDense over
// Float32Rows, leaves the source as it was, and does not see later writes
// to it.
func sameAsNewDense[S, T vector.Scalar](t *testing.T, src vector.Slice[S], qs [][]float32) {
	t.Helper()
	src.Data = slices.Clone(src.Data)
	before := slices.Clone(src.Data)
	got, err := DenseEncoder[S, T]{}.Encode(src)
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
	want, err := NewDense[T](rows)
	if err != nil {
		t.Fatal(err)
	}
	clear(src.Data)
	if got.Len() != want.Len() || got.Dims() != want.Dims() {
		t.Fatalf("shape %dx%d, NewDense %dx%d", got.Len(), got.Dims(), want.Len(), want.Dims())
	}
	a, b := make([]float32, got.Len()), make([]float32, got.Len())
	for _, q := range qs {
		scoreL2(t, got, q, a)
		scoreL2(t, want, q, b)
		for i := range a {
			if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
				t.Fatalf("%T to %T row %d: Encode scores %v, NewDense %v", src.Data, got, i, a[i], b[i])
			}
		}
	}
}

// otherEmbedding is a representation Dense and Block do not score.
type otherEmbedding struct{}

func (otherEmbedding) Dims() int                                { return 8 }
func (otherEmbedding) Score(Embedding, Metric) (float32, error) { return 0, nil }

// typedQueries are one query per source scalar, each with the values it
// should decode to, known without Dense: halves through the float16
// package (as numbers, not bits), int8 and uint8 rows that span their whole
// code range so the block scale is 1, int8 negative and uint8 past 127.
func typedQueries(t *testing.T) ([]Embedding, [][]float64) {
	f32 := []float32{0.25, -1, 3, 7, -30, 2, 0.5, -0.75}
	f16 := []float32{1.0 / 3, -2, 0.1, 100, 2048.5, -0.5, 7, 1e-3}
	f16Want := make([]float64, len(f16))
	for i, x := range f16 {
		f16Want[i] = float64(float16.Fromfloat32(x).Float32())
	}
	i8 := []float32{-128, 127, -3, 5, 0, -1, 100, -100}
	u8 := []float32{0, 255, 128, 200, 1, 127, 129, 64}
	at := func(e Embedding, err error) Embedding {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
		return e
	}
	a, err1 := NewDense[float32]([][]float32{f32})
	b, err2 := NewDense[float16.Float16]([][]float32{f16})
	c, err3 := NewDense[int8]([][]float32{i8})
	d, err4 := NewDense[uint8]([][]float32{u8})
	qs := []Embedding{at(a.At(0), err1), at(b.At(0), err2), at(c.At(0), err3), at(d.At(0), err4)}
	return qs, [][]float64{f64s(f32), f16Want, f64s(i8), f64s(u8)}
}

func f64s(v []float32) []float64 {
	out := make([]float64, len(v))
	for i, x := range v {
		out[i] = float64(x)
	}
	return out
}

// exactScore is metric's score of q against r in float64.
func exactScore(metric Metric, q, r []float64) float64 {
	var dot, qq, rr, l2 float64
	for j := range q {
		dot, qq, rr = dot+q[j]*r[j], qq+q[j]*q[j], rr+r[j]*r[j]
		l2 += (q[j] - r[j]) * (q[j] - r[j])
	}
	switch metric {
	case L2:
		return -l2
	case Cosine:
		if qq == 0 || rr == 0 {
			return 0
		}
		return dot / math.Sqrt(qq*rr)
	}
	return dot
}

// Every Dense target type scores every dense query type under every metric
// as the float64 score of the values both decode to (not the values before
// encoding), At(i).Score agrees with ScoreAll, ScoreAll writes only
// dst[:Len()], and neither changes the query.
func TestDenseScoresTypedQueries(t *testing.T) {
	f := [][]float32{{0.1, -2.5, 1000, 3.75, -7, 0, 0.5, 2e-3}, {1.0 / 3, 65504, -1, 1, 2, 3, 4, 5}, make([]float32, 8)}
	fWant := [][]float64{f64s(f[0]), f64s(f[1]), f64s(f[2])}
	hWant := make([][]float64, len(f))
	for i, r := range f {
		hWant[i] = make([]float64, len(r))
		for j, x := range r {
			hWant[i][j] = float64(float16.Fromfloat32(x).Float32())
		}
	}
	i8 := [][]float32{{-128, 127, -3, 5, 0, -1, 100, -100}, {7, -7, 1, 2, 3, 4, 5, 6}, make([]float32, 8)}
	u8 := [][]float32{{0, 255, 128, 200, 1, 127, 129, 64}, {7, 8, 9, 10, 250, 251, 252, 3}, make([]float32, 8)}
	must := func(b EmbeddingBatch, err error) EmbeddingBatch {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
		return b
	}
	targets := []EmbeddingBatch{must(NewDense[float32](f)), must(NewDense[float16.Float16](f)), must(NewDense[int8](i8)), must(NewDense[uint8](u8))}
	values := [][][]float64{fWant, hWant, {f64s(i8[0]), f64s(i8[1]), f64s(i8[2])}, {f64s(u8[0]), f64s(u8[1]), f64s(u8[2])}}
	qs, qWant := typedQueries(t)
	for ti, b := range targets {
		for qi, q := range qs {
			before := make([]float32, 8)
			if err := q.(interface{ Decode([]float32) error }).Decode(before); err != nil {
				t.Fatal(err)
			}
			for _, metric := range []Metric{L2, Cosine, InnerProduct} {
				dst := []float32{9, 9, 9, 9, 9}
				if err := b.ScoreAll(q, metric, dst); err != nil {
					t.Fatalf("%T %T %v: %v", b, q, metric, err)
				}
				if dst[3] != 9 || dst[4] != 9 {
					t.Fatalf("%T: ScoreAll wrote past Len: %v", b, dst)
				}
				for i := range b.Len() {
					want := exactScore(metric, qWant[qi], values[ti][i])
					mag := 1.0
					if metric != Cosine {
						mag = -exactScore(L2, qWant[qi], make([]float64, 8)) - exactScore(L2, values[ti][i], make([]float64, 8))
					}
					if d := math.Abs(float64(dst[i]) - want); !(d <= 1e-6*mag) {
						t.Fatalf("%T row %d, %T query, %v: %v, want %v", b, i, q, metric, dst[i], want)
					}
					one, err := b.At(i).Score(q, metric)
					if err != nil || math.Abs(float64(one-dst[i])) > 1e-6*mag {
						t.Fatalf("%T row %d %v: Score %v (%v), ScoreAll %v", b, i, metric, one, err, dst[i])
					}
				}
			}
			after := make([]float32, 8)
			if err := q.(interface{ Decode([]float32) error }).Decode(after); err != nil || !slices.Equal(after, before) {
				t.Fatalf("%T query changed: %v, was %v", q, after, before)
			}
		}
	}
}

// Decode copies exactly the values a block scores: a float32 row is a copy,
// not a view; int8 and uint8 rows are offset + scale*code, within half a
// step of the input. It leaves dst's tail alone and rejects a short dst.
func TestDenseEmbeddingDecode(t *testing.T) {
	rows := [][]float32{{1, 2, 3, 4}, {-0.5, 0.25, 9, -8}}
	f, err := NewDense[float32](rows)
	if err != nil {
		t.Fatal(err)
	}
	dst := []float32{0, 0, 0, 0, 7}
	if err := f.At(1).(DenseEmbedding[float32]).Decode(dst); err != nil || !slices.Equal(dst, []float32{-0.5, 0.25, 9, -8, 7}) {
		t.Fatalf("Decode: %v, %v", dst, err)
	}
	clear(dst)
	again := make([]float32, 4)
	if err := f.At(1).(DenseEmbedding[float32]).Decode(again); err != nil || !slices.Equal(again, rows[1]) {
		t.Fatalf("writing Decode's output changed the block: %v", again)
	}
	if err := f.At(0).(DenseEmbedding[float32]).Decode(make([]float32, 3)); err == nil {
		t.Fatal("a 3-wide dst was accepted")
	}

	rng := rand.New(rand.NewSource(5))
	narrow := make([][]float32, 5)
	for i := range narrow {
		narrow[i] = make([]float32, 16)
		for j := range narrow[i] {
			narrow[i][j] = float32(-3 + 0.7*rng.Float64())
		}
	}
	checkAffine[int8](t, narrow)
	checkAffine[uint8](t, narrow)
}

// checkAffine: each decoded coordinate is within half a step of its input,
// and the block scores exactly as a Dense[float32] of the decoded rows.
func checkAffine[T int8 | uint8](t *testing.T, rows [][]float32) {
	t.Helper()
	d, err := NewDense[T](rows)
	if err != nil {
		t.Fatal(err)
	}
	lo, hi := math.Inf(1), math.Inf(-1)
	for _, r := range rows {
		for _, x := range r {
			lo, hi = min(lo, float64(x)), max(hi, float64(x))
		}
	}
	half := (hi - lo) / 255 / 2 * 1.0001
	decoded := make([][]float32, len(rows))
	for i, r := range rows {
		decoded[i] = make([]float32, len(r))
		if err := d.At(i).(DenseEmbedding[T]).Decode(decoded[i]); err != nil {
			t.Fatal(err)
		}
		for j, x := range r {
			if math.Abs(float64(decoded[i][j])-float64(x)) > half {
				t.Fatalf("%T row %d coord %d: %v decodes to %v", d, i, j, x, decoded[i][j])
			}
		}
	}
	f, err := NewDense[float32](decoded)
	if err != nil {
		t.Fatal(err)
	}
	q := dq(t, []float32{1, -2, 0.5, 3, 0, 0, 1, 1, -1, 2, 2, 2, 0.25, -0.25, 9, -9})
	for _, metric := range []Metric{L2, Cosine, InnerProduct} {
		a, b := make([]float32, len(rows)), make([]float32, len(rows))
		if err := d.ScoreAll(q, metric, a); err != nil {
			t.Fatal(err)
		}
		if err := f.ScoreAll(q, metric, b); err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(a, b) {
			t.Fatalf("%T %v: %v, Dense[float32] of the decoded rows %v", d, metric, a, b)
		}
	}
}

// Scoring fails, rather than writing scores, for a query of another width
// or representation, an unknown metric, a short dst or an infinite score;
// At panics outside the rows.
func TestDenseScoreErrors(t *testing.T) {
	d, err := NewDense[int8]([][]float32{{1, 2, 3, 4, 5, 6, 7, 8}, {-1, 0, 1, 0, -1, 0, 1, 0}})
	if err != nil {
		t.Fatal(err)
	}
	q := dq(t, []float32{1, 1, 1, 1, 1, 1, 1, 1})
	ok := make([]float32, 2)
	for name, err := range map[string]error{
		"7-wide query":   d.ScoreAll(dq(t, make([]float32, 7)), L2, ok),
		"nil query":      d.ScoreAll(nil, L2, ok),
		"typed nil":      d.ScoreAll((*DenseEmbedding[float32])(nil), L2, ok),
		"foreign query":  d.ScoreAll(otherEmbedding{}, L2, ok),
		"unknown metric": d.ScoreAll(q, Metric(9), ok),
		"short dst":      d.ScoreAll(q, L2, ok[:1]),
		"zero Dense":     Dense[float32]{}.ScoreAll(q, L2, ok),
		"zero view":      func() error { _, err := DenseEmbedding[uint8]{}.Score(q, L2); return err }(),
		"row foreign":    func() error { _, err := d.At(0).Score(otherEmbedding{}, L2); return err }(),
	} {
		if err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
	far, err := NewDense[float32]([][]float32{{1e19}})
	if err != nil {
		t.Fatal(err)
	}
	if err := far.ScoreAll(dq(t, []float32{-1e19}), L2, ok); err == nil {
		t.Errorf("an L2 score past float32 (%v) was accepted", ok[0])
	}
	for _, i := range []int{-1, 2} {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("At(%d) of 2 rows did not panic", i)
				}
			}()
			d.At(i)
		}()
	}
}

// Concurrent ScoreAll calls on one block, each with its own dst, all get
// the serial scores.
func TestDenseConcurrentScoreAll(t *testing.T) {
	d, err := NewDense[float16.Float16](treeRand(500, 32, 3))
	if err != nil {
		t.Fatal(err)
	}
	qs := treeRand(16, 32, 4)
	want := make([][]float32, len(qs))
	for i, q := range qs {
		want[i] = make([]float32, d.Len())
		scoreL2(t, d, q, want[i])
	}
	var wg sync.WaitGroup
	for range 8 {
		wg.Go(func() {
			dst := make([]float32, d.Len())
			for i, q := range qs {
				if err := d.ScoreAll(dq(t, q), L2, dst); err != nil || !slices.Equal(dst, want[i]) {
					t.Error("a concurrent ScoreAll disagreed")
					return
				}
			}
		})
	}
	wg.Wait()
}

// BenchmarkDenseScoreAll is one dense query against a 256 x 128 block of
// each type, the per-node work of a tree search.
func BenchmarkDenseScoreAll(b *testing.B) {
	rows := treeRand(256, 128, 1)
	q := dq(b, treeRand(1, 128, 2)[0])
	dst := make([]float32, len(rows))
	bench := func(name string, d EmbeddingBatch, err error) {
		if err != nil {
			b.Fatal(err)
		}
		for _, metric := range []Metric{L2, Cosine} {
			b.Run(name+"/"+metric.String(), func(b *testing.B) {
				b.ReportAllocs()
				for b.Loop() {
					if err := d.ScoreAll(q, metric, dst); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
	f32, err := NewDense[float32](rows)
	bench("float32", f32, err)
	f16, err := NewDense[float16.Float16](rows)
	bench("float16", f16, err)
	i8, err := NewDense[int8](rows)
	bench("int8", i8, err)
	u8, err := NewDense[uint8](rows)
	bench("uint8", u8, err)
}

// Cosine of tiny nonzero vectors, whose float32 squared norms and dot
// products underflow to 0, is their direction's cosine, from ScoreAll and
// from Score; only a true zero scores 0. An affine int8 block of tiny
// values scores the cosine of its decoded values, computed here in float64.
func TestDenseCosineTinyVectors(t *testing.T) {
	rows := [][]float32{{1e-30, 0}, {-1e-30, 0}, {0, 1e-30}, {0, 0}, {3e-25, 4e-25}}
	f, err := NewDense[float32](rows)
	must(t, err)
	checkCosine(t, f, []float64{1, -1, 0, 0, 0.6})
	i8, err := NewDense[int8](rows[:4])
	must(t, err)
	var want []float64
	q := make([]float32, 2)
	must(t, i8.At(0).(DenseEmbedding[int8]).Decode(q))
	for i := range i8.Len() {
		r := make([]float32, 2)
		must(t, i8.At(i).(DenseEmbedding[int8]).Decode(r))
		want = append(want, exactScore(Cosine, f64s(q), f64s(r)))
	}
	if want[0] < 0.99 || want[1] > -0.99 {
		t.Fatalf("int8 decoded rows lost their direction: %v", want)
	}
	checkCosine(t, i8, want)
	z, err := NewDense[float32]([][]float32{{0, 0}, {1e-30, 0}})
	must(t, err)
	got := []float32{9, 9}
	if err := z.ScoreAll(z.At(0), Cosine, got); err != nil || got[0] != 0 || got[1] != 0 {
		t.Fatalf("a zero query: %v, %v", got, err)
	}
}

func checkCosine(t *testing.T, d EmbeddingBatch, want []float64) {
	t.Helper()
	q := d.At(0)
	got := make([]float32, d.Len())
	must(t, d.ScoreAll(q, Cosine, got))
	for i := range want {
		one, err := d.At(i).Score(q, Cosine)
		if err != nil || math.Abs(float64(got[i])-want[i]) > 1e-6 || one != got[i] {
			t.Fatalf("%T row %d: ScoreAll %v, Score %v (%v), want %v", d, i, got[i], one, err, want[i])
		}
	}
}

func must(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}
