package ivfq

import "github.com/axiomhq/ivfq/internal/simd"

// Metric is the distance a namespace scores with.
type Metric int

const (
	L2 Metric = iota
	Cosine
	InnerProduct
)

// Dot returns the inner product. With precomputed squared norms it gives a
// squared distance without the sqrt L2Sq pays: |a-b|^2 = |a|^2 + |b|^2 - 2 a.b.
// The subtraction loses a few low bits when a and b are much longer than
// their difference, so callers rank with it; they do not report it.
func Dot(a, b []float32) float32 { return simd.Dot(a, b) }

// L2Sq returns squared euclidean distance. Squared preserves ranking and
// skips the sqrt.
func L2Sq(a, b []float32) float32 { return simd.L2Sq(a, b) }

// CosineSim returns cosine similarity (higher = more similar). Zero vectors
// score 0.
func CosineSim(a, b []float32) float32 { return simd.CosineSim(a, b) }

// Normalize scales v to unit length in place. A zero-norm v is left as is.
func Normalize(v []float32) { simd.Normalize(v) }

// Score converts a metric into a uniform "higher = better" score.
func Score(metric string, q, v []float32) float32 {
	if metric == "l2" {
		return -L2Sq(q, v)
	}
	return CosineSim(q, v)
}

// Dots sets out[r] to Dot(q, block[r*len(q):(r+1)*len(q)]) for every row
// of block, bit for bit, in one call.
func Dots(q, block, out []float32) { simd.Dots(q, block, out) }
