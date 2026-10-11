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

// The quantizer kernels equal their Go references bit for bit (the block
// kernels also equal a plain per-row loop), across widths and tails.
func TestQuantKernels(t *testing.T) {
	r := rand.New(rand.NewPCG(7, 8))
	for _, dims := range []int{1, 3, 4, 5, 31, 32, 33, 63, 64, 65, 100, 128, 768, 1000} {
		block := make([]float32, dims*FillBlock)
		for i := range block {
			block[i] = float32(r.NormFloat64())
			if i%17 == 0 {
				block[i] = 0
			}
			if i%53 == 0 {
				block[i] = float32(math.Copysign(0, -1)) // -0 >= 0 is true
			}
		}
		var sums, want [FillBlock]float64
		BlockSumSq(block, dims, &sums)
		for j := 0; j < dims; j++ {
			for k := range want {
				x := float64(block[j*FillBlock+k])
				want[k] += x * x
			}
		}
		if sums != want {
			t.Fatalf("dims=%d BlockSumSq differs", dims)
		}

		var abs, wabs [FillBlock]float64
		words := make([]uint64, (dims+63)/64*FillBlock)
		BlockSignAbs(block, dims, &abs, words)
		wwords := make([]uint64, len(words))
		for j := 0; j < dims; j++ {
			for k := 0; k < FillBlock; k++ {
				v := block[j*FillBlock+k]
				if v >= 0 {
					wwords[(j/64)*FillBlock+k] |= 1 << (j % 64)
				}
				wabs[k] += math.Abs(float64(v))
			}
		}
		if abs != wabs || !slices.Equal(words, wwords) {
			t.Fatalf("dims=%d BlockSignAbs differs", dims)
		}

		x := block[:dims]
		for _, mul := range []float64{1, 0.37} {
			lo, hi := MinMaxScaled(x, mul)
			wlo, whi := math.Inf(1), math.Inf(-1)
			for _, v := range x {
				wlo, whi = min(wlo, float64(v)*mul), max(whi, float64(v)*mul)
			}
			if lo != wlo || hi != whi {
				t.Fatalf("dims=%d MinMaxScaled %v %v, want %v %v", dims, lo, hi, wlo, whi)
			}
			delta := (hi - lo) / 15
			if delta <= 0 {
				continue // QuantCodes needs delta > 0
			}
			codes, wcodes := make([]byte, dims), make([]byte, dims)
			got := QuantCodes(codes, x, mul, lo, delta, 15)
			hasFWHTOld := hasFWHT
			hasFWHT = false
			wgot := QuantCodes(wcodes, x, mul, lo, delta, 15)
			hasFWHT = hasFWHTOld
			if got != wgot || !slices.Equal(codes, wcodes) {
				t.Fatalf("dims=%d mul=%v QuantCodes vector %v, reference %v", dims, mul, got, wgot)
			}
		}
		sq := SumSq(x)
		hasFWHTOld := hasFWHT
		hasFWHT = false
		wsq := SumSq(x)
		hasFWHT = hasFWHTOld
		if sq != wsq {
			t.Fatalf("dims=%d SumSq %v, reference %v", dims, sq, wsq)
		}
	}
}
