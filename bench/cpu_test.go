package bench

import "testing"

func TestCPUSeconds(t *testing.T) {
	before := CPUSeconds()
	x := 0
	for i := 0; i < 50_000_000; i++ {
		x += i & 7
	}
	if after := CPUSeconds(); after < before || x < 0 {
		t.Fatalf("CPUSeconds went backwards: %v -> %v", before, after)
	}
}
