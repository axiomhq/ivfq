package ivfq

import "github.com/go-simd/floats"

// The two kernels below are the hottest scalar code: the index build
// is ~94% L2Sq and every query scores its candidates through
// one of them. Their bodies are github.com/go-simd/floats — pure Go,
// CGO_ENABLED=0, generated SIMD assembly for arm64/amd64 (and five more) with
// a scalar fallback everywhere else, one transitive dep (golang.org/x/sys).
// Measured at 128 dims on an M3 Max: L2Sq 99.1 -> 41.7 ns/call (2.4x),
// CosineSim 98.7 -> 64.9 ns/call (1.5x).
//
// It changes summation order: the kernels keep several accumulator lanes and
// accumulate float32 (the old loops widened every element to float64), so
// results move by ~1e-7 relative. Rankings are unaffected at that magnitude
// and nothing in the tree asserts an exact distance — but k-means centroids
// are only reproducible against a FIXED kernel, so treat a kernel swap the
// way you would treat a seed change (assignAll says the same).

// CosineSim returns cosine similarity (higher = more similar). Zero vectors score 0.
func CosineSim(a, b []float32) float32 {
	// floats returns NaN for a zero-magnitude input (its only NaN source
	// for finite input); that is this function's documented 0.
	if s := floats.Float32CosineSimilarity(a, b); s == s {
		return s
	}
	return 0
}

// Dot returns the inner product. With precomputed squared norms it gives a
// squared distance without the sqrt L2Sq pays: |a-b|^2 = |a|^2 + |b|^2 - 2 a.b.
// The subtraction loses a few low bits when a and b are much longer than
// their difference, so callers rank with it; they do not report it.
func Dot(a, b []float32) float32 { return floats.Float32Dot(a, b) }

// L2Sq returns squared euclidean distance. Squared preserves ranking and skips the sqrt.
// (floats has no squared form, so the sqrt is paid and undone — still 2.4x
// the scalar loop it replaced.)
func L2Sq(a, b []float32) float32 {
	d := floats.Float32Distance(a, b)
	return d * d
}

// Score converts a metric into a uniform "higher = better" score.
func Score(metric string, q, v []float32) float32 {
	if metric == "l2" {
		return -L2Sq(q, v)
	}
	return CosineSim(q, v)
}
