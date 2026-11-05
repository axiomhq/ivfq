package simd

import "math"

// roundI8 is byte(int8(max(-128, min(127, math.Round(x))))) without
// math.Round: clamping first leaves a value whose float64 sum with ±0.5 is
// exact, so truncating it toward zero rounds half away from zero. An
// integer already in range takes the first branch and is its own rounding.
func roundI8(x float32) byte {
	if i := int32(x); float32(i) == x && i >= -128 && i <= 127 {
		return byte(int8(i))
	}
	z := max(-128, min(127, float64(x)))
	return byte(int8(int32(z + math.Copysign(0.5, z))))
}

// DecodeI8 widens each byte of b to float32 in dst; eight lanes at a time with AVX2.
func DecodeI8(dst []float32, b []byte) {
	i := 0
	if hasAVX2 && len(dst) >= 8 {
		i = len(dst) &^ 7
		_ = b[len(dst)-1]
		decodeI8AVX2(&dst[0], &b[0], i)
	}
	for ; i < len(dst); i++ {
		dst[i] = float32(int8(b[i]))
	}
}

// AppendI8 appends each value of v rounded to int8. The vector kernel takes the leading run of
// eight-lane groups that are all integers — every row of an i8 column,
// which a rewrite re-encodes — where rounding is the identity and
// clamping is int8 saturation; roundI8 takes the rest.
func AppendI8(dst []byte, v []float32) []byte {
	at := len(dst)
	dst = append(dst, make([]byte, len(v))...)
	out := dst[at:]
	i := 0
	if hasAVX2 && len(v) >= 8 {
		i = encodeI8AVX2(&out[0], &v[0], len(v)&^7)
	}
	for ; i < len(v); i++ {
		out[i] = roundI8(v[i])
	}
	return dst
}
