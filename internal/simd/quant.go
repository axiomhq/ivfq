package simd

import (
	"math"
	"math/bits"
)

// Kernels behind rabitq's encoder and query quantizer. The block kernels run
// FillBlock rows as independent lanes, so each row's float64 sum adds its
// terms in dimension order, exactly as a scalar loop over that row would. The
// query kernels sum in four interleaved lanes, combined (l0+l2)+(l1+l3), with
// the tail after; the Go references use the same order, so a result does not
// depend on the CPU. No quant kernel fuses a multiply into an add, so the
// references round every product with an explicit float64 conversion, which
// stops the compiler fusing them (it does on arm64 and GOAMD64=v3). The
// squares of a float32 are exact in float64 and need it only for uniformity;
// the products in QuantCodes do need it.

// BlockSumSq adds, for each of the FillBlock rows r, the squares of its
// first dims values to sums[r] (float64 of each float32, in dimension order).
// block is dimension-major: block[j*FillBlock+r].
func BlockSumSq(block []float32, dims int, sums *[FillBlock]float64) {
	_ = block[dims*FillBlock-1]
	if hasAVX2FMA && dims > 0 {
		blockSumSq(&block[0], dims, &sums[0])
		return
	}
	for j := 0; j < dims; j++ {
		for r, v := range block[j*FillBlock:][:FillBlock] {
			x := float64(v)
			sums[r] += float64(x * x)
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
	if hasAVX2FMA && n > 0 {
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
	if hasAVX2FMA && len(x) >= 4 {
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
	if hasAVX2FMA && n4 > 0 {
		s = sumSq(&x[0], n4)
	} else {
		var l [4]float64
		for j := 0; j < n4; j++ {
			f := float64(x[j])
			l[j&3] += float64(f * f)
		}
		s = (l[0] + l[2]) + (l[1] + l[3])
	}
	for _, v := range x[n4:] {
		f := float64(v)
		s += float64(f * f)
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
	if hasAVX2FMA && n4 > 0 {
		s = quantCodes(&codes[0], &x[0], n4, mul, lo, delta, float64(levels))
	} else {
		var l [4]float64
		for j := 0; j < n4; j++ {
			c, e := quantOne(float64(float64(x[j])*mul), lo, delta, levels)
			codes[j], l[j&3] = c, l[j&3]+float64(e*e)
		}
		s = (l[0] + l[2]) + (l[1] + l[3])
	}
	for j := n4; j < len(x); j++ {
		c, e := quantOne(float64(float64(x[j])*mul), lo, delta, levels)
		codes[j] = c
		s += float64(e * e)
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
	return byte(t), lo + float64(delta*t) - x
}

// ScoreFast is rabitq's scoreFast row loop for its first len(scores)&^3
// rows, four rows to a vector, and returns how many rows it set (0 without
// AVX2); the caller's Go loop sets the rest. Per row r, with k = delta, low,
// tail, scale, qNorm, qNorm²: zeroRowAlign rows (align -1) score 0;
// otherwise dist2 = norm²+q2, less 2·norm·qn·ip where norm, qn and align
// are positive, ip = (2·(delta·sums[r] + low·ones[r]) - tail)·scale/align,
// then max(dist2, (norm-qn)²), scored -dist2 (l2) or min(1, max(-1,
// 1-dist2/2)). Each operation is the Go loop's, unfused, in its order, so
// the scores are its bits. qn must be positive.
func ScoreFast(sums, ones []uint16, norms, aligns []float32, k *[6]float64, l2 bool, scores []float32) int {
	n := len(scores) &^ 3
	if !hasAVX2 || n == 0 {
		return 0
	}
	_, _, _, _ = sums[n-1], ones[n-1], norms[n-1], aligns[n-1]
	var cos int
	if !l2 {
		cos = 1
	}
	scoreFast4(n, &sums[0], &ones[0], &norms[0], &aligns[0], &scores[0], &k[0], cos)
	return n
}

// SubSumSq sets x[j] -= c[j] in float32 and returns SumSq of the result,
// bit for bit, in one pass. c must be as long as x.
func SubSumSq(x, c []float32) float64 {
	_ = c[len(x)-1]
	n8 := len(x) &^ 7
	if !hasAVX2FMA || n8 == 0 {
		for j := range x {
			x[j] -= c[j]
		}
		return SumSq(x)
	}
	var lanes [4]float64
	subSumSq(&x[0], &c[0], n8, &lanes)
	for j := n8; j < len(x); j++ {
		x[j] -= c[j]
	}
	n4 := len(x) &^ 3
	for j := n8; j < n4; j++ {
		f := float64(x[j])
		lanes[j&3] += float64(f * f)
	}
	s := (lanes[0] + lanes[2]) + (lanes[1] + lanes[3])
	for _, v := range x[n4:] {
		f := float64(v)
		s += float64(f * f)
	}
	return s
}

// Planes sets bit j%64 of planes[p*words + j/64] to bit p of codes[j]
// (codes below 16, planes cleared by the caller) and returns Σ codes[j].
func Planes(codes []byte, planes []uint64, words int) uint64 {
	_ = planes[4*words-1]
	j, sum := 0, uint64(0)
	if hasAVX2 && len(codes) >= 32 {
		j = len(codes) &^ 31
		sum = planes32(&codes[0], j, &planes[0], words)
	}
	for ; j < len(codes); j++ {
		c, w, sh := uint64(codes[j]), j/64, uint(j%64)
		sum += c
		for p := range 4 {
			planes[p*words+w] |= (c >> p & 1) << sh
		}
	}
	return sum
}

var planesLUTTable = func() [256]byte {
	var t [256]byte
	for n := range 16 {
		for m := range 16 {
			t[n*16+m] = byte(bits.OnesCount8(uint8(n & m)))
		}
	}
	return t
}()

// PlanesLUT builds the 16-byte FastScan subset-sum lookup tables for groups
// groups of 4 dimensions from the 4 bit planes (planes has 4*words uint64s,
// lut has at least 16*groups bytes). Bit for bit identical to the Go loop.
func PlanesLUT(planes []uint64, words int, lut []byte, groups int) {
	if groups <= 0 {
		return
	}
	_ = planes[4*words-1]
	_ = lut[16*groups-1]
	g := 0
	if hasAVX2 && groups >= 2 {
		pairs := groups / 2
		planesLUTAVX2(&planes[0], words, &lut[0], pairs, &planesLUTTable[0])
		g = pairs * 2
	}
	for ; g < groups; g++ {
		w, sh := g/16, uint((g%16)*4)
		n0 := (planes[w] >> sh) & 0xF
		n1 := (planes[words+w] >> sh) & 0xF
		n2 := (planes[2*words+w] >> sh) & 0xF
		n3 := (planes[3*words+w] >> sh) & 0xF
		t := lut[16*g : 16*(g+1)]
		for m := range 16 {
			t[m] = planesLUTTable[n0*16+uint64(m)] +
				planesLUTTable[n1*16+uint64(m)]<<1 +
				planesLUTTable[n2*16+uint64(m)]<<2 +
				planesLUTTable[n3*16+uint64(m)]<<3
		}
	}
}

// Unpack1Bit sets dst[j] = base[j] + (±s) where bit j of words selects +s (1)
// or -s (0), bit for bit identical to the Go loop. base and words must cover
// len(dst).
func Unpack1Bit(dst, base []float32, words []uint64, s float32) {
	n := len(dst)
	if n == 0 {
		return
	}
	_ = base[n-1]
	_ = words[(n-1)>>6]
	j := 0
	if hasAVX2 && n >= 8 {
		j = n &^ 7
		unpack1BitAVX2(&dst[0], &base[0], &words[0], j, s)
	}
	pm := [2]float32{-s, s}
	for ; j < n; j++ {
		dst[j] = base[j] + pm[words[j>>6]>>(uint(j)&63)&1]
	}
}

// SubStoreSumSq sets dst[j] = a[j] - b[j] in float32 and returns SumSq(dst)
// in one pass. a and b must be at least len(dst) long.
func SubStoreSumSq(dst, a, b []float32) float64 {
	n := len(dst)
	if n == 0 {
		return 0
	}
	_, _ = a[n-1], b[n-1]
	n8 := n &^ 7
	if !hasAVX2FMA || n8 == 0 {
		for j := range dst {
			dst[j] = a[j] - b[j]
		}
		return SumSq(dst)
	}
	var lanes [4]float64
	subStoreSumSq(&dst[0], &a[0], &b[0], n8, &lanes)
	for j := n8; j < n; j++ {
		dst[j] = a[j] - b[j]
	}
	n4 := n &^ 3
	for j := n8; j < n4; j++ {
		f := float64(dst[j])
		lanes[j&3] += float64(f * f)
	}
	s := (lanes[0] + lanes[2]) + (lanes[1] + lanes[3])
	for _, v := range dst[n4:] {
		f := float64(v)
		s += float64(f * f)
	}
	return s
}

// SignAbs sets bit j of words to 1 iff x[j] >= 0 (words must be zeroed and
// cover len(x)) and returns Σ |float64(x[j])| in the four-lane order.
func SignAbs(x []float32, words []uint64) float64 {
	n := len(x)
	if n == 0 {
		return 0
	}
	_ = words[(n-1)>>6]
	n8 := n &^ 7
	var s float64
	if hasAVX2 && n8 > 0 {
		var lanes [4]float64
		signAbsAVX2(&x[0], n8, &words[0], &lanes)
		n4 := n &^ 3
		for j := n8; j < n4; j++ {
			if x[j] >= 0 {
				words[j>>6] |= 1 << (uint(j) & 63)
			}
			lanes[j&3] += math.Abs(float64(x[j]))
		}
		s = (lanes[0] + lanes[2]) + (lanes[1] + lanes[3])
		for j := n4; j < n; j++ {
			if x[j] >= 0 {
				words[j>>6] |= 1 << (uint(j) & 63)
			}
			s += math.Abs(float64(x[j]))
		}
		return s
	}
	for j, v := range x {
		if v >= 0 {
			words[j>>6] |= 1 << (uint(j) & 63)
		}
		s += math.Abs(float64(v))
	}
	return s
}

// CodesLUT builds the 16-byte FastScan subset-sum lookup tables directly from
// codes (each code < 16) into lut (at least 16*((len(codes)+3)/4) bytes).
// Bit for bit identical to PlanesLUT.
func CodesLUT(codes, lut []byte) {
	n := len(codes)
	if n == 0 {
		return
	}
	groups := (n + 3) / 4
	_ = lut[16*groups-1]
	j := 0
	if hasAVX2 && n >= 8 {
		n8 := n &^ 7
		codesLUTAVX2(&codes[0], &lut[0], n8>>3)
		j = n8
	}
	for g := j / 4; g < groups; g++ {
		var c [4]byte
		for b := range 4 {
			if g*4+b < n {
				c[b] = codes[g*4+b]
			}
		}
		t := lut[16*g : 16*(g+1)]
		for m := range 16 {
			var s byte
			if m&1 != 0 {
				s += c[0]
			}
			if m&2 != 0 {
				s += c[1]
			}
			if m&4 != 0 {
				s += c[2]
			}
			if m&8 != 0 {
				s += c[3]
			}
			t[m] = s
		}
	}
}
