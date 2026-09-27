package ivfq

import "testing"

// TestDefaultNprobeCurve pins the whole default across five orders of
// magnitude, because it is three rules interacting (a constant budget, a
// k/32 recall floor, a k/4 ceiling) and the interesting values are the
// crossovers, not any one rule.
func TestDefaultNprobeCurve(t *testing.T) {
	for _, tc := range []struct{ total, k, want int }{
		{64, 8, 8},                 // ceiling floors at 16, clamped to k -> exact
		{256, 16, 16},              // exact
		{20_000, 20, 16},           // ceiling's 16 first bites here
		{100_000, 98, 24},          // ceiling: k/4, the case this clamp exists for
		{400_000, 391, 97},         // ceiling and budget within one of each other
		{1_000_000, 977, 98},       // budget
		{10_000_000, 9766, 305},    // recall floor, k/32
		{100_000_000, 97657, 3051}, // recall floor
	} {
		if got := HoodK(tc.total); got != tc.k {
			t.Fatalf("HoodK(%d) = %d, want %d", tc.total, got, tc.k)
		}
		got := min(DefaultNprobe(tc.k), tc.k)
		if got != tc.want {
			t.Errorf("N=%d k=%d: default nprobe %d, want %d", tc.total, tc.k, got, tc.want)
		}
		// The two invariants the three rules must never break.
		if got > tc.k {
			t.Errorf("N=%d: default %d exceeds k=%d", tc.total, got, tc.k)
		}
		if tc.k >= 64 && got > tc.k/4 {
			t.Errorf("N=%d: default %d is more than a quarter of k=%d", tc.total, got, tc.k)
		}
	}
}
