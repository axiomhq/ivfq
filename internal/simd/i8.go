package simd

import "math"

// RoundI8 is byte(int8(max(-128, min(127, math.Round(x))))) without
// math.Round: clamping first leaves a value whose float64 sum with ±0.5 is
// exact, so truncating it toward zero rounds half away from zero. An
// integer already in range takes the first branch and is its own rounding.
func RoundI8(x float32) byte {
	if i := int32(x); float32(i) == x && i >= -128 && i <= 127 {
		return byte(int8(i))
	}
	z := max(-128, min(127, float64(x)))
	return byte(int8(int32(z + math.Copysign(0.5, z))))
}
