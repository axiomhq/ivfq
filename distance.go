package ivfq

import (
	"fmt"

	"github.com/axiomhq/ivfq/internal/simd"
)

// Metric is the distance a field is scored with. The zero value is L2.
type Metric int

const (
	L2           Metric = iota // squared euclidean distance, negated by Score
	Cosine                     // cosine similarity
	InnerProduct               // inner product
)

var metricNames = [...]string{L2: "l2", Cosine: "cosine", InnerProduct: "ip"}

// ParseMetric is the inverse of Metric.String: "l2", "cosine" or "ip".
func ParseMetric(s string) (Metric, error) {
	for m, name := range metricNames {
		if name == s {
			return Metric(m), nil
		}
	}
	return 0, fmt.Errorf("ivfq: unknown metric %q", s)
}

func (m Metric) String() string {
	if m < 0 || int(m) >= len(metricNames) {
		return fmt.Sprintf("Metric(%d)", int(m))
	}
	return metricNames[m]
}

// MarshalText implements encoding.TextMarshaler, so a Metric reads and
// writes as its name in JSON and other text configs.
func (m Metric) MarshalText() ([]byte, error) {
	if m < 0 || int(m) >= len(metricNames) {
		return nil, fmt.Errorf("ivfq: unknown metric %d", int(m))
	}
	return []byte(metricNames[m]), nil
}

// UnmarshalText implements encoding.TextUnmarshaler.
func (m *Metric) UnmarshalText(text []byte) error {
	parsed, err := ParseMetric(string(text))
	if err != nil {
		return err
	}
	*m = parsed
	return nil
}

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

// Score is q against v under metric as a "higher is better" score: the
// negated squared distance for L2, the similarity for Cosine, the inner
// product for InnerProduct. An unknown Metric is a programming error and
// panics.
func Score(metric Metric, q, v []float32) float32 {
	switch metric {
	case L2:
		return -L2Sq(q, v)
	case Cosine:
		return CosineSim(q, v)
	case InnerProduct:
		return Dot(q, v)
	}
	panic("ivfq: Score: unknown " + metric.String())
}

// Dots sets out[r] to Dot(q, block[r*len(q):(r+1)*len(q)]) for every row
// of block, bit for bit, in one call.
func Dots(q, block, out []float32) { simd.Dots(q, block, out) }
