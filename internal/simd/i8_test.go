package simd

import (
	"math"
	"testing"
)

// scalarI8 is RoundI8's definition: round half away from zero, saturate.
func scalarI8(x float32) byte {
	return byte(int8(max(-128, min(127, math.Round(float64(x))))))
}

// RoundI8 on integers, which the vector kernel takes instead on a host
// with AVX2: the scalar fallback must agree with the definition too.
func TestRoundI8ScalarOnIntegers(t *testing.T) {
	for i := -70000; i <= 70000; i++ {
		x := float32(i)
		if got, want := RoundI8(x), scalarI8(x); got != want {
			t.Fatalf("RoundI8(%v) = %d, want %d", x, int8(got), int8(want))
		}
	}
	for _, x := range []float32{2147483648, -2147483648, -2147483904, 1e30, -1e30, float32(math.Copysign(0, -1)), 0.5, -0.5, 2.5, -2.5} {
		if got, want := RoundI8(x), scalarI8(x); got != want {
			t.Fatalf("RoundI8(%v) = %d, want %d", x, int8(got), int8(want))
		}
	}
}
