package simd

import (
	"math"
	"math/rand/v2"
	"slices"
	"testing"
)

// TestNormalizeTiny: the scale 1/|v| for a vector near the float32 floor
// exceeds MaxFloat32, so it must be applied in float64 or the result is Inf.
func TestNormalizeTiny(t *testing.T) {
	v := []float32{math.SmallestNonzeroFloat32, 0}
	Normalize(v)
	if v[0] != 1 || v[1] != 0 {
		t.Fatalf("Normalize = %v, want [1 0]", v)
	}
	z := []float32{0, 0}
	Normalize(z)
	if z[0] != 0 || z[1] != 0 {
		t.Fatalf("zero vector changed: %v", z)
	}
}

// Dot and L2Sq agree with a float64 reference at every tail length, with
// the AVX-512 kernels and without.
func TestF32Kernels(t *testing.T) {
	r := rand.New(rand.NewPCG(9, 10))
	old := hasAVX512F
	defer func() { hasAVX512F = old }()
	for _, on := range []bool{true, false} {
		hasAVX512F = old && on
		if got := L2Sq([]float32{0, 0}, []float32{3, 3}); got != 18 {
			t.Fatalf("avx512=%v: L2Sq([0 0], [3 3]) = %v, want 18", hasAVX512F, got)
		}
		for _, n := range []int{1, 2, 15, 16, 17, 31, 48, 63, 64, 65, 79, 80, 128, 129, 255, 256, 257, 768, 1537} {
			a, b := make([]float32, n), make([]float32, n)
			var dot, l2, magD, magL float64
			for i := range a {
				a[i], b[i] = float32(r.NormFloat64()*10), float32(r.NormFloat64()*10)
				p, d := float64(a[i])*float64(b[i]), float64(a[i])-float64(b[i])
				dot, l2, magD, magL = dot+p, l2+d*d, magD+math.Abs(p), magL+d*d
			}
			if got := float64(Dot(a, b)); math.Abs(got-dot) > 1e-5*magD {
				t.Fatalf("avx512=%v n=%d: Dot %v, want %v", hasAVX512F, n, got, dot)
			}
			if got := float64(L2Sq(a, b)); math.Abs(got-l2) > 1e-5*magL {
				t.Fatalf("avx512=%v n=%d: L2Sq %v, want %v", hasAVX512F, n, got, l2)
			}
		}
	}
}

// L2Sq has one 64-lane order in Go and SIMD, including unaligned tails.
func TestL2SqLanes(t *testing.T) {
	r := rand.New(rand.NewPCG(23, 24))
	old512, old2 := hasAVX512F, hasAVX2
	defer func() { hasAVX512F, hasAVX2 = old512, old2 }()
	for _, n := range []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 767, 768, 769} {
		for pattern := range 2 {
			a, b := make([]float32, n+1)[1:], make([]float32, n+1)[1:]
			for i := range a {
				a[i], b[i] = float32(r.NormFloat64()*10), float32(r.NormFloat64()*10)
				if pattern == 1 {
					a[i], b[i] = 1e-5, 0
				}
			}
			if pattern == 1 && n >= 2 {
				a[0], a[1] = 4096, 1 // Tiny terms must collect before the final reduction.
			}
			var lanes [64]float32
			for lane := range lanes {
				for i := lane; i < n; i += 64 {
					d := a[i] - b[i]
					lanes[lane] += float32(d * d)
				}
			}
			// Fold adjacent 16-lane quarters first, then halves of each width.
			for width := 32; width >= 16; width /= 2 {
				for base := 0; base < width; base += 16 {
					for j := range 16 {
						lanes[base+j] = lanes[2*base+j] + lanes[2*base+j+16]
					}
				}
			}
			for width := 8; width > 0; width /= 2 {
				for j := range width {
					lanes[j] += lanes[j+width]
				}
			}
			want := lanes[0]
			for mode := range 3 {
				hasAVX512F, hasAVX2 = old512 && mode == 0, old2 && mode < 2
				if got := L2Sq(a, b); math.Float32bits(got) != math.Float32bits(want) {
					t.Fatalf("mode=%d n=%d pattern=%d: L2Sq = %v, want %v", mode, n, pattern, got, want)
				}
			}
		}
	}
}

// A broken length precondition panics with and without the vector kernels,
// instead of reading or writing outside the slices. The short slices keep
// spare capacity, which a slice-expression guard would accept.
func TestShapeMismatchPanics(t *testing.T) {
	old := hasAVX512F
	defer func() { hasAVX512F = old }()
	for _, on := range []bool{true, false} {
		hasAVX512F = old && on
		for name, f := range map[string]func(){
			"Dot":             func() { Dot(make([]float32, 16), make([]float32, 17)) },
			"L2Sq":            func() { L2Sq(make([]float32, 16), make([]float32, 17)) },
			"L2SqShort":       func() { L2Sq(make([]float32, 17), make([]float32, 16, 17)) },
			"L2SqEmpty":       func() { L2Sq(nil, []float32{1}) },
			"FWHT":            func() { FWHT(make([]float32, 24)) },
			"FWHTRows":        func() { FWHTRows(make([]float32, 3*FillBlock), 3) },
			"DecodeI8":        func() { DecodeI8(make([]float32, 16), make([]byte, 8, 16)) },
			"BitProductBytes": func() { BitProductBytes(make([]byte, 8, 16), make([]uint64, 8), 2) },
		} {
			func() {
				defer func() {
					if recover() == nil {
						t.Errorf("avx512=%v: %s with a bad shape did not panic", hasAVX512F, name)
					}
				}()
				f()
			}()
		}
	}
}

// CosineSim scores 0, never NaN or ±Inf, when a norm is 0, underflows or
// overflows float32.
func TestCosineSimDegenerate(t *testing.T) {
	for _, c := range []struct{ a, b []float32 }{
		{[]float32{0, 0}, []float32{1, 0}},
		{[]float32{1e-30, 0}, []float32{1, 0}},
		{[]float32{1e20, 1e20}, []float32{1e20, 1e20}},
		{[]float32{1e20, 1e20}, []float32{1, 1}},
	} {
		if got := CosineSim(c.a, c.b); got != 0 {
			t.Errorf("CosineSim(%v, %v) = %v, want 0", c.a, c.b, got)
		}
	}
	if got := CosineSim([]float32{3, 4}, []float32{6, 8}); math.Abs(float64(got)-1) > 1e-6 {
		t.Errorf("CosineSim of parallel vectors = %v, want 1", got)
	}
}

// NonFinite, AddF64, AddF32 and ScaleF64 match their Go loops bit for bit at every
// tail length, and NonFinite finds the first bad value wherever it sits.
func TestF32xKernels(t *testing.T) {
	r := rand.New(rand.NewPCG(11, 12))
	old := hasAVX2
	defer func() { hasAVX2 = old }()
	for _, n := range []int{0, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 768, 771} {
		x := make([]float32, n)
		for i := range x {
			x[i] = float32(r.NormFloat64() * math.Exp(r.NormFloat64()*20))
		}
		if n > 1 {
			x[0], x[n-1] = math.SmallestNonzeroFloat32, -math.MaxFloat32 // finite extremes
		}
		s := 1 / math.Sqrt(r.Float64()*1e3+1e-30)
		var want [2][]float32
		var sums [2][]float64
		for k, on := range []bool{true, false} {
			hasAVX2 = old && on
			if got := NonFinite(x); got != -1 {
				t.Fatalf("avx2=%v n=%d: NonFinite = %d on finite values", hasAVX2, n, got)
			}
			for _, bad := range []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))} {
				for _, at := range []int{0, n / 2, n - 1} {
					if at < 0 || at >= n {
						continue
					}
					y := append([]float32(nil), x...)
					y[at] = bad
					if n > 1 && at < n-1 {
						y[n-1] = float32(math.NaN()) // a later bad value must not win
					}
					if got := NonFinite(y); got != at {
						t.Fatalf("avx2=%v n=%d: NonFinite = %d, want %d", hasAVX2, n, got, at)
					}
				}
			}
			sums[k] = make([]float64, n+1)
			sums[k][n] = 7
			for i := range sums[k][:n] {
				sums[k][i] = r.NormFloat64()
			}
			if k == 1 {
				copy(sums[1], sums[0]) // same start for both
			}
			want[k] = append([]float32(nil), x...)
			ScaleF64(want[k], s)
		}
		// Recompute AddF64 from equal starts, then compare with a Go loop.
		base := append([]float64(nil), sums[0]...)
		for k, on := range []bool{true, false} {
			hasAVX2 = old && on
			copy(sums[k], base)
			AddF64(sums[k], x)
		}
		for j := range x {
			ref := base[j] + float64(x[j])
			if math.Float64bits(sums[0][j]) != math.Float64bits(ref) || math.Float64bits(sums[1][j]) != math.Float64bits(ref) {
				t.Fatalf("n=%d j=%d: AddF64 %v %v, want %v", n, j, sums[0][j], sums[1][j], ref)
			}
			ref32 := float32(float64(x[j]) * s)
			if math.Float32bits(want[0][j]) != math.Float32bits(ref32) || math.Float32bits(want[1][j]) != math.Float32bits(ref32) {
				t.Fatalf("n=%d j=%d: ScaleF64 %v %v, want %v", n, j, want[0][j], want[1][j], ref32)
			}
		}
		if sums[0][n] != 7 || sums[1][n] != 7 {
			t.Fatalf("n=%d: AddF64 wrote past len(src)", n)
		}
		start := make([]float32, n+1)
		for i := range start[:n] {
			start[i] = float32(r.NormFloat64() * 1e3)
		}
		start[n] = 7
		var adds [2][]float32
		for k, on := range []bool{true, false} {
			hasAVX2 = old && on
			adds[k] = append([]float32(nil), start...)
			AddF32(adds[k], x)
		}
		for j := range x {
			ref := start[j] + x[j]
			if math.Float32bits(adds[0][j]) != math.Float32bits(ref) || math.Float32bits(adds[1][j]) != math.Float32bits(ref) {
				t.Fatalf("n=%d j=%d: AddF32 %v %v, want %v", n, j, adds[0][j], adds[1][j], ref)
			}
		}
		if adds[0][n] != 7 || adds[1][n] != 7 {
			t.Fatalf("n=%d: AddF32 wrote past len(src)", n)
		}
	}
}

// ScoreFast is its Go loop bit for bit, at both metrics, on zero rows
// (align -1), rows at the centroid (norm 0), unaligned rows (align 0),
// rows whose norm gap bounds the distance, and any row count.
func TestScoreFastMatchesLoop(t *testing.T) {
	r := rand.New(rand.NewPCG(9, 9))
	for _, n := range []int{0, 3, 4, 5, 8, 33, 530} {
		sums, ones := make([]uint16, n), make([]uint16, n)
		norms, aligns := make([]float32, n), make([]float32, n)
		for i := range n {
			sums[i], ones[i] = uint16(r.IntN(3000)), uint16(r.IntN(768))
			norms[i], aligns[i] = float32(r.Float64()*2), float32(0.5+r.Float64()*0.4)
			switch r.IntN(8) {
			case 0:
				aligns[i] = -1
			case 1:
				norms[i] = 0
			case 2:
				aligns[i] = 0
			case 3:
				norms[i] = 40 // far: the gap bounds it
			}
		}
		qn := 0.3 + r.Float64()
		k := [6]float64{r.Float64() * 0.01, -r.Float64() * 0.05, r.Float64() * 3, 1 / math.Sqrt(768), qn, qn * qn}
		for _, l2 := range []bool{true, false} {
			got := make([]float32, n)
			done := ScoreFast(sums, ones, norms, aligns, &k, l2, got)
			if hasAVX2 && done != n&^3 {
				t.Fatalf("n=%d: set %d rows", n, done)
			}
			for i := range done {
				var want float32
				align := float64(aligns[i])
				if align != -1 {
					norm := float64(norms[i])
					dist2 := norm*norm + k[5]
					if norm > 0 && align > 0 {
						dot := 2*(k[0]*float64(sums[i])+k[1]*float64(ones[i])) - k[2]
						ip := dot * k[3] / align
						dist2 -= 2 * norm * k[4] * ip
					}
					gap := norm - k[4]
					dist2 = max(dist2, gap*gap)
					if l2 {
						want = float32(-dist2)
					} else {
						want = float32(min(1, max(-1, 1-dist2/2)))
					}
				}
				if math.Float32bits(got[i]) != math.Float32bits(want) {
					t.Fatalf("n=%d l2=%v row %d: %v, want %v", n, l2, i, got[i], want)
				}
			}
		}
	}
}

// SubSumSq is the subtraction then SumSq, bit for bit; Planes is the
// bit-per-plane layout and the code sum, at every tail.
func TestSubSumSqAndPlanes(t *testing.T) {
	r := rand.New(rand.NewPCG(5, 1))
	for _, n := range []int{1, 3, 4, 7, 8, 9, 31, 32, 33, 64, 100, 768, 1000} {
		x, c := make([]float32, n), make([]float32, n)
		for i := range x {
			x[i], c[i] = float32(r.NormFloat64()*1e3), float32(r.NormFloat64()*1e3)
		}
		want := make([]float32, n)
		for i := range want {
			want[i] = x[i] - c[i]
		}
		ws := SumSq(want)
		got := SubSumSq(x, c)
		if math.Float64bits(got) != math.Float64bits(ws) || !slices.Equal(x, want) {
			t.Fatalf("n=%d: SubSumSq %v, want %v", n, got, ws)
		}
		codes := make([]byte, n)
		var sum uint64
		for i := range codes {
			codes[i] = byte(r.IntN(16))
			sum += uint64(codes[i])
		}
		words := (n + 63) / 64
		planes := make([]uint64, 4*words)
		if s := Planes(codes, planes, words); s != sum {
			t.Fatalf("n=%d: sum %d, want %d", n, s, sum)
		}
		for j, cd := range codes {
			for p := range 4 {
				if got := planes[p*words+j/64] >> (j % 64) & 1; got != uint64(cd>>p&1) {
					t.Fatalf("n=%d dim %d plane %d: %d", n, j, p, got)
				}
			}
		}
		for p := range 4 { // no bit past n
			if n%64 != 0 && planes[p*words+words-1]>>(n%64) != 0 {
				t.Fatalf("n=%d plane %d: bits past the end", n, p)
			}
		}
	}
}

func TestPlanesLUTAndUnpack1Bit(t *testing.T) {
	r := rand.New(rand.NewPCG(9, 2))
	for _, n := range []int{1, 3, 4, 7, 8, 9, 15, 16, 31, 32, 64, 100, 768} {
		codes := make([]byte, n)
		for i := range codes {
			codes[i] = byte(r.IntN(16))
		}
		words := (n + 63) / 64
		planes := make([]uint64, 4*words)
		Planes(codes, planes, words)
		groups := (n + 3) / 4
		got := make([]byte, 16*groups)
		PlanesLUT(planes, words, got, groups)
		gotCodes := make([]byte, 16*groups)
		CodesLUT(codes, gotCodes)
		if !slices.Equal(got, gotCodes) {
			t.Fatalf("n=%d: CodesLUT != PlanesLUT", n)
		}
		for g := range groups {
			for m := range 16 {
				var want byte
				for b := range 4 {
					if m>>b&1 != 0 && g*4+b < n {
						want += codes[g*4+b]
					}
				}
				if got[16*g+m] != want {
					t.Fatalf("n=%d group=%d m=%d: got %d, want %d", n, g, m, got[16*g+m], want)
				}
			}
		}

		base := make([]float32, n)
		for i := range base {
			base[i] = float32(r.NormFloat64())
		}
		bitsW := make([]uint64, words)
		for i := range bitsW {
			bitsW[i] = r.Uint64()
		}
		s := float32(0.375)
		dst := make([]float32, n)
		Unpack1Bit(dst, base, bitsW, s)
		pm := [2]float32{-s, s}
		for j := range n {
			want := base[j] + pm[bitsW[j>>6]>>(uint(j)&63)&1]
			if math.Float32bits(dst[j]) != math.Float32bits(want) {
				t.Fatalf("n=%d j=%d: got %v, want %v", n, j, dst[j], want)
			}
		}

		// SubStoreSumSq and SignAbs
		a, b := make([]float32, n), make([]float32, n)
		for i := range a {
			a[i], b[i] = float32(r.NormFloat64()*10), float32(r.NormFloat64()*10)
		}
		if n > 0 {
			a[0] = float32(math.Copysign(0, -1)) // -0.0 must set sign bit 1 (>= 0)
			b[0] = 0
		}
		dst2 := make([]float32, n)
		gotSS := SubStoreSumSq(dst2, a, b)
		wantDst := make([]float32, n)
		for i := range a {
			wantDst[i] = a[i] - b[i]
		}
		wantSS := SumSq(wantDst)
		if math.Float64bits(gotSS) != math.Float64bits(wantSS) || !slices.Equal(dst2, wantDst) {
			t.Fatalf("n=%d: SubStoreSumSq %v, want %v", n, gotSS, wantSS)
		}
		gotW := make([]uint64, words)
		gotAbs := SignAbs(a, gotW)
		wantW := make([]uint64, words)
		var wantAbs float64
		for j, v := range a {
			if v >= 0 {
				wantW[j>>6] |= 1 << (uint(j) & 63)
			}
			wantAbs += math.Abs(float64(v))
		}
		if !slices.Equal(gotW, wantW) || math.Abs(gotAbs-wantAbs) > 1e-10*(1+wantAbs) {
			t.Fatalf("n=%d: SignAbs words/abs mismatch: got %v, want %v", n, gotAbs, wantAbs)
		}
	}
}
