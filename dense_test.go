package ivfq

import (
	"math"
	"math/rand"
	"slices"
	"testing"

	"github.com/axiomhq/ivfq/vector"
	"github.com/x448/float16"
)

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
	d.ScoreAll(q, scores)
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
		exact.ScoreAll(q, got)
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
		got.ScoreAll(q, a)
		want.ScoreAll(q, b)
		for i := range a {
			if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
				t.Fatalf("%T to %T row %d: Encode scores %v, NewDense %v", src.Data, got, i, a[i], b[i])
			}
		}
	}
}
