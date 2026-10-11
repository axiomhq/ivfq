package simd

import "math"

// Kernels behind rabitq's encoder and query quantizer. The block kernels run
// FillBlock rows as independent lanes, so each row's float64 sum adds its
// terms in dimension order, exactly as a scalar loop over that row would. The
// query kernels sum in four interleaved lanes, combined (l0+l2)+(l1+l3), with
// the tail after; the Go references use the same order, so a result does not
// depend on the CPU.

// BlockSumSq adds, for each of the FillBlock rows r, the squares of its
// first dims values to sums[r] (float64 of each float32, in dimension order).
// block is dimension-major: block[j*FillBlock+r].
func BlockSumSq(block []float32, dims int, sums *[FillBlock]float64) {
	_ = block[dims*FillBlock-1]
	if hasFWHT && dims > 0 {
		blockSumSq(&block[0], dims, &sums[0])
		return
	}
	for j := 0; j < dims; j++ {
		for r, v := range block[j*FillBlock:][:FillBlock] {
			x := float64(v)
			sums[r] += x * x
		}
	}
}

// BlockSignAbs sets bit j%64 of words[(j/64)*FillBlock+r] when row r's value
// at dimension j is >= 0 (words is dims/64 rounded up groups of FillBlock,
// each word written whole), and adds |value| to abs[r] in dimension order.
func BlockSignAbs(block []float32, dims int, abs *[FillBlock]float64, words []uint64) {
	_ = block[dims*FillBlock-1]
	_ = words[((dims+63)/64)*FillBlock-1]
	for g := 0; g*64 < dims; g++ {
		var lo, hi [FillBlock]uint32
		n := min(64, dims-g*64)
		blockBits32(block[g*64*FillBlock:], min(n, 32), abs, &lo)
		if n > 32 {
			blockBits32(block[(g*64+32)*FillBlock:], n-32, abs, &hi)
		}
		for r := range lo {
			words[g*FillBlock+r] = uint64(lo[r]) | uint64(hi[r])<<32
		}
	}
}

// blockBits32 is BlockSignAbs for n <= 32 dimensions: out[r] bit j is row r's
// sign at dimension j.
func blockBits32(block []float32, n int, abs *[FillBlock]float64, out *[FillBlock]uint32) {
	if hasFWHT && n > 0 {
		blockBits(&block[0], n, &abs[0], &out[0])
		return
	}
	*out = [FillBlock]uint32{}
	for j := 0; j < n; j++ {
		for r, v := range block[j*FillBlock:][:FillBlock] {
			if v >= 0 {
				out[r] |= 1 << j
			}
			abs[r] += math.Abs(float64(v))
		}
	}
}

// MinMaxScaled returns the smallest and largest float64(x[j])*mul. x must
// not be empty.
func MinMaxScaled(x []float32, mul float64) (lo, hi float64) {
	lo, hi = math.Inf(1), math.Inf(-1)
	i := 0
	if hasFWHT && len(x) >= 4 {
		i = len(x) &^ 3
		lo, hi = minMaxScaled(&x[0], i, mul)
	}
	for ; i < len(x); i++ {
		v := float64(x[i]) * mul
		lo, hi = min(lo, v), max(hi, v)
	}
	return lo, hi
}

// SumSq returns the sum of float64(x[j])^2, in the lane order above.
func SumSq(x []float32) float64 {
	n4 := len(x) &^ 3
	var s float64
	if hasFWHT && n4 > 0 {
		s = sumSq(&x[0], n4)
	} else {
		var l [4]float64
		for j := 0; j < n4; j++ {
			f := float64(x[j])
			l[j&3] += f * f
		}
		s = (l[0] + l[2]) + (l[1] + l[3])
	}
	for _, v := range x[n4:] {
		f := float64(v)
		s += f * f
	}
	return s
}

// QuantCodes sets codes[j] to the uniform scalar quantization of
// x[j]*mul over [lo, lo+delta*levels] (nearest, ties up; levels <= 255,
// delta > 0, x[j]*mul >= lo) and returns the squared rounding error summed
// in the lane order above.
func QuantCodes(codes []byte, x []float32, mul, lo, delta float64, levels int) float64 {
	n4 := len(x) &^ 3
	_ = codes[len(x)-1]
	var s float64
	if hasFWHT && n4 > 0 {
		s = quantCodes(&codes[0], &x[0], n4, mul, lo, delta, float64(levels))
	} else {
		var l [4]float64
		for j := 0; j < n4; j++ {
			c, e := quantOne(float64(x[j])*mul, lo, delta, levels)
			codes[j], l[j&3] = c, l[j&3]+e*e
		}
		s = (l[0] + l[2]) + (l[1] + l[3])
	}
	for j := n4; j < len(x); j++ {
		c, e := quantOne(float64(x[j])*mul, lo, delta, levels)
		codes[j] = c
		s += e * e
	}
	return s
}

// quantOne is one lane of QuantCodes. y >= 0, so Trunc is the cheap rounding:
// y - Trunc(y) is exact.
func quantOne(x, lo, delta float64, levels int) (code byte, err float64) {
	y := (x - lo) / delta
	t := math.Trunc(y)
	if y-t >= 0.5 {
		t++
	}
	t = min(float64(levels), max(0, t))
	return byte(t), lo + delta*t - x
}
