package simd

import (
	"math"
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
