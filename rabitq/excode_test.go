package rabitq

import (
	"math"
	"math/rand"
	"slices"
	"testing"
)

// exCodeReference is exCode as it was before it ran a plane word at a time
// (6d14b35c): the encoder every stored ex code was written by.
func exCodeReference(x []float32, bits int, t float64, planes []uint64) (yNorm, align float32) {
	w := bitWords(len(x))
	var n2 float64
	for _, v := range x {
		n2 += float64(v) * float64(v)
	}
	if n2 == 0 {
		return 0, 0
	}
	n := math.Sqrt(n2)
	t /= n
	top := 1<<bits - 1
	var dot, y2 float64
	for j, v := range x {
		a := math.Abs(float64(v))
		e := min(int(t*a), top)
		low := e
		if v < 0 {
			low = top - e
		}
		for p := range bits {
			planes[p*w+j/64] |= uint64(low>>p&1) << uint(j%64)
		}
		y := float64(e) + 0.5
		dot += a * y
		y2 += y * y
	}
	yn := math.Sqrt(y2)
	return float32(yn), float32(min(1, dot/(n*yn)))
}

// TestExCodeMatchesReference: exCode sets the same planes and returns the
// same two float32 bits as the reference, at every ExBits, on widths on and
// off a word boundary, for random rows and for rows built to hit its edges.
func TestExCodeMatchesReference(t *testing.T) {
	r := rand.New(rand.NewSource(1))
	negZero := float32(math.Copysign(0, -1))
	for _, dims := range []int{1, 3, 7, 8, 9, 63, 64, 65, 100, 127, 128, 129, 200, 768, 1000} {
		for bits := 1; bits <= maxExBits; bits++ {
			edges := map[string][]float32{
				"zero":        make([]float32, dims),
				"negzero":     slices.Repeat([]float32{negZero}, dims),
				"one":         onehot(dims, dims-1, 1),
				"minusone":    onehot(dims, dims/2, -3),
				"ties":        signed(dims, func(int) float32 { return 1 }),                // every |x_j| equal, n = sqrt(dims)
				"steps":       signed(dims, func(j int) float32 { return float32(j % 9) }), // many equal magnitudes and zeros
				"huge":        signed(dims, func(j int) float32 { return math.MaxFloat32 / float32(j+2) }),
				"tiny":        signed(dims, func(j int) float32 { return math.SmallestNonzeroFloat32 * float32(j%5+1) }),
				"subnormal1":  onehot(dims, 0, math.SmallestNonzeroFloat32),
				"max1":        onehot(dims, dims-1, -math.MaxFloat32),
				"mixedscales": signed(dims, func(j int) float32 { return float32(math.Ldexp(1, j%80-40)) }),
			}
			for _, j := range []int{0, dims / 3, dims - 1} {
				edges["negzero"][j] = 0.5 // -0 beside nonzero values: -0 is not < 0
			}
			scales := []float64{exScale(dims, bits), 1, float64(int(1) << bits), 1e-3, 1e6}
			// Cuts exactly at integers: the "ties" row has ||x|| = sqrt(dims), so
			// t = k*sqrt(dims) puts t|x_j|/||x|| at k when the sqrt is exact.
			if s := math.Sqrt(float64(dims)); s == math.Floor(s) {
				for k := range 1<<bits + 1 {
					scales = append(scales, float64(k)*s)
				}
			}
			check := func(name string, x []float32, scale float64) {
				t.Helper()
				w := bitWords(dims)
				want, got := make([]uint64, bits*w), make([]uint64, bits*w)
				wn, wa := exCodeReference(x, bits, scale, want)
				gn, ga := exCode(x, bits, scale, got)
				if math.Float32bits(wn) != math.Float32bits(gn) || math.Float32bits(wa) != math.Float32bits(ga) || !slices.Equal(want, got) {
					t.Fatalf("dims %d bits %d %s t=%v: got (%v %v %x), want (%v %v %x)", dims, bits, name, scale, gn, ga, got, wn, wa, want)
				}
			}
			for name, x := range edges {
				for _, s := range scales {
					check(name, x, s)
				}
			}
			for i := range 50 {
				x := make([]float32, dims)
				for j := range x {
					x[j] = float32(r.NormFloat64())
				}
				check("random", x, scales[i%len(scales)])
			}
		}
	}
}

func onehot(dims, j int, v float32) []float32 {
	x := make([]float32, dims)
	x[j] = v
	return x
}

// signed is f(j) with alternating, irregular signs.
func signed(dims int, f func(int) float32) []float32 {
	x := make([]float32, dims)
	for j := range x {
		x[j] = f(j)
		if j%3 == 1 || j%7 == 0 {
			x[j] = -x[j]
		}
	}
	return x
}

func BenchmarkExCode(b *testing.B) {
	const dims, bits = 768, 3
	r := rand.New(rand.NewSource(1))
	xs := make([][]float32, 256)
	for i := range xs {
		xs[i] = make([]float32, dims)
		for j := range xs[i] {
			xs[i][j] = float32(r.NormFloat64())
		}
	}
	t, planes := exScale(dims, bits), make([]uint64, bits*bitWords(dims))
	for _, f := range []struct {
		name string
		fn   func([]float32, int, float64, []uint64) (float32, float32)
	}{{"new", exCode}, {"reference", exCodeReference}} {
		b.Run(f.name, func(b *testing.B) {
			for i := 0; b.Loop(); i++ {
				clear(planes)
				f.fn(xs[i%len(xs)], bits, t, planes)
			}
		})
	}
}
