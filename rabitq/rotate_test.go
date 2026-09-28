package rabitq

import (
	"math"
	"testing"
)

func TestSeedNeverZero(t *testing.T) {
	if Seed() == 0 {
		t.Fatal("empty seed")
	}
	if Seed("ns", "vector") == 0 {
		t.Fatal("named seed")
	}
}

func TestRotationPreservesNorm(t *testing.T) {
	v := []float32{1, 2, 3, 4, 5}
	var n0 float64
	for _, x := range v {
		n0 += float64(x) * float64(x)
	}
	NewRotation(Seed("t", "r"), len(v)).Apply(v)
	var n1 float64
	for _, x := range v {
		n1 += float64(x) * float64(x)
	}
	if math.Abs(n0-n1) > 1e-5 {
		t.Fatalf("norm %v -> %v", n0, n1)
	}
}
