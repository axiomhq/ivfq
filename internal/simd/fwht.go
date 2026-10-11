package simd

// FillBlock is how many rows the *Rows kernels transform together: a
// dimension-major block holds one dimension's FillBlock row values per
// 16 consecutive float32. The vector kernels are written for exactly this.
const FillBlock = 16

// The kernels behind the rabitq rotation: a fast Walsh-Hadamard transform
// and a scaled gather. Each has a Go reference and an AVX2+FMA kernel that
// produces the same bits: the transform does the same additions in the same
// stage order (an in-register stage is x*±1 + neighbour, exact), the gather
// is one multiply per element.

// FWHT transforms x in place by the unnormalised Walsh-Hadamard matrix; len(x)
// must be a power of two. Dividing by sqrt(len(x)) makes it orthogonal.
func FWHT(x []float32) {
	n := len(x)
	if n >= 8 && hasFWHT {
		fwht8(&x[0], n)
		if n > 8 {
			fwhtStages(&x[0], n, 8)
		}
		return
	}
	fwhtGeneric(x)
}

// FWHTRows is FWHT across n rows of FillBlock values: x[j*FillBlock+r] is
// element j of row r, and each row is transformed independently.
func FWHTRows(x []float32, n int) {
	if len(x) != n*FillBlock {
		panic("simd: FWHTRows block of the wrong size")
	}
	if n > 1 && hasFWHT {
		fwhtStages(&x[0], n*FillBlock, FillBlock)
		return
	}
	fwhtRowsGeneric(x, n)
}

// PermScale sets dst[i] = coef[i] * src[perm[i]]. perm must be a permutation
// of 0..len(dst)-1 (any in-range indexes are memory safe); all four are as
// long as dst, and dst must not overlap src.
func PermScale(dst, src []float32, perm []int32, coef []float32) {
	n := len(dst)
	_, _, _ = perm[n-1], coef[n-1], src[n-1]
	i := 0
	if hasFWHT && n >= 4 {
		i = n &^ 3
		permScale(&dst[0], &src[0], &perm[0], &coef[0], i)
	}
	for ; i < n; i++ {
		dst[i] = coef[i] * src[perm[i]]
	}
}

// Scale sets dst[i] = coef[i] * src[i]; the three are as long as dst, and
// dst may be src.
func Scale(dst, src, coef []float32) {
	n := len(dst)
	_, _ = src[n-1], coef[n-1]
	i := 0
	if hasFWHT && n >= 8 {
		i = n &^ 7
		scale(&dst[0], &src[0], &coef[0], i)
	}
	for ; i < n; i++ {
		dst[i] = coef[i] * src[i]
	}
}

// PermScaleRows is PermScale over rows of FillBlock values:
// row i of dst is coef[i] times row perm[i] of src.
func PermScaleRows(dst, src []float32, perm []int32, coef []float32) {
	n := len(perm)
	_, _, _ = dst[n*FillBlock-1], src[n*FillBlock-1], coef[n-1]
	if hasFWHT && n > 0 {
		permScaleRows(&dst[0], &src[0], &perm[0], &coef[0], n)
		return
	}
	for i, p := range perm {
		d, s := dst[i*FillBlock:][:FillBlock], src[int(p)*FillBlock:][:FillBlock]
		for k := range d {
			d[k] = coef[i] * s[k]
		}
	}
}

func fwhtGeneric(x []float32) {
	for h := 1; h < len(x); h <<= 1 {
		for base := 0; base < len(x); base += 2 * h {
			for i := base; i < base+h; i++ {
				a, b := x[i], x[i+h]
				x[i], x[i+h] = a+b, a-b
			}
		}
	}
}

func fwhtRowsGeneric(x []float32, n int) {
	for h := 1; h < n; h <<= 1 {
		for base := 0; base < n; base += 2 * h {
			for i := base; i < base+h; i++ {
				ri, rj := x[i*FillBlock:][:FillBlock], x[(i+h)*FillBlock:][:FillBlock]
				for k := range ri {
					a, b := ri[k], rj[k]
					ri[k], rj[k] = a+b, a-b
				}
			}
		}
	}
}
