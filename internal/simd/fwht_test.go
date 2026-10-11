package simd

import (
	"math"
	"math/rand/v2"
	"slices"
	"testing"
)

// Every kernel is its Go reference, bit for bit, at each power-of-two
// width, and FWHT is the Hadamard matrix (checked against the definition).
func TestFWHTMatchesGeneric(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	for _, n := range []int{1, 2, 4, 8, 16, 32, 64, 256, 1024, 2048} {
		x := make([]float32, n)
		for i := range x {
			x[i] = float32(r.NormFloat64())
		}
		got, want := slices.Clone(x), slices.Clone(x)
		FWHT(got)
		fwhtGeneric(want)
		for i := range got {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("n=%d i=%d: %v, want %v", n, i, got[i], want[i])
			}
		}
		if n <= 64 { // H[i][j] = (-1)^popcount(i&j)
			for i := range x {
				var s float64
				for j, v := range x {
					s += float64(v) * float64(1-2*(popcount(i&j)&1))
				}
				if math.Abs(s-float64(got[i])) > 1e-4*float64(n) {
					t.Fatalf("n=%d row %d: %v, want %v", n, i, got[i], s)
				}
			}
		}
	}
}

func popcount(x int) (n int) {
	for ; x != 0; x &= x - 1 {
		n++
	}
	return
}

func TestFWHTRowsMatchesGeneric(t *testing.T) {
	r := rand.New(rand.NewPCG(3, 4))
	for _, n := range []int{1, 2, 4, 8, 64, 512} {
		x := make([]float32, n*FillBlock)
		for i := range x {
			x[i] = float32(r.NormFloat64())
		}
		got, want := slices.Clone(x), slices.Clone(x)
		FWHTRows(got, n)
		fwhtRowsGeneric(want, n)
		if !slices.Equal(got, want) {
			t.Fatalf("n=%d differs from the reference", n)
		}
		// Each row alone is FWHT of its column.
		for k := 0; k < FillBlock; k++ {
			col := make([]float32, n)
			for j := range col {
				col[j] = x[j*FillBlock+k]
			}
			FWHT(col)
			for j := range col {
				if math.Float32bits(col[j]) != math.Float32bits(got[j*FillBlock+k]) {
					t.Fatalf("n=%d row %d elem %d", n, k, j)
				}
			}
		}
	}
}

func TestPermScaleMatchesGeneric(t *testing.T) {
	r := rand.New(rand.NewPCG(5, 6))
	for _, n := range []int{1, 7, 8, 9, 100, 1536} {
		src := make([]float32, n)
		coef := make([]float32, n)
		for i := range src {
			src[i], coef[i] = float32(r.NormFloat64()), float32(r.NormFloat64())
		}
		perm := make([]int32, n)
		for i, p := range r.Perm(n) {
			perm[i] = int32(p)
		}
		got, want := make([]float32, n), make([]float32, n)
		PermScale(got, src, perm, coef)
		for i := range want {
			want[i] = coef[i] * src[perm[i]]
		}
		if !slices.Equal(got, want) {
			t.Fatalf("n=%d differs", n)
		}

		sc, wsc := make([]float32, n), make([]float32, n)
		Scale(sc, src, coef)
		for i := range wsc {
			wsc[i] = coef[i] * src[i]
		}
		if !slices.Equal(sc, wsc) {
			t.Fatalf("Scale n=%d differs", n)
		}

		rows := make([]float32, n*FillBlock)
		for i := range rows {
			rows[i] = float32(r.NormFloat64())
		}
		g, w := make([]float32, len(rows)), make([]float32, len(rows))
		PermScaleRows(g, rows, perm, coef)
		for i, p := range perm {
			for k := 0; k < FillBlock; k++ {
				w[i*FillBlock+k] = coef[i] * rows[int(p)*FillBlock+k]
			}
		}
		if !slices.Equal(g, w) {
			t.Fatalf("rows n=%d differs", n)
		}
	}
}
