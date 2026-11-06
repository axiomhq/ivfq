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
