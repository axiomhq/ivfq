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

// Embedding is one vector in some representation: a row of a Dense block
// (DenseEmbedding), of a rabitq.Block, or of any other EmbeddingBatch.
//
// Score is query scored against the receiver, the stored target, under
// metric, higher is better: the negated squared distance for L2 (up to
// rounding, so a row equal to the query may score just above 0), the
// similarity for Cosine (0 when either vector is zero), the inner product
// for InnerProduct. A representation may score an estimate (RaBitQ does).
// It returns an error for a query of another width, a query
// representation or metric the receiver does not score, or a non-finite
// result. Every receiver in this module takes a dense query (DenseEmbedding
// of any scalar, the value At returns, not a pointer to it) and no other;
// none asks the query to score itself instead.
// Score neither changes nor retains query.
type Embedding interface {
	Dims() int
	Score(query Embedding, metric Metric) (float32, error)
}

// EmbeddingBatch is Len embeddings of one width held together, scored
// together.
//
// At returns row i as an Embedding: a read-only view that keeps the batch
// (and how to read it) alive, never shared scratch. It panics, as slice
// indexing does, when i is not a row.
//
// ScoreAll sets dst[i], for every i < Len(), in row order, to what
// At(i).Score(query, metric) returns, up to float rounding, and never
// writes dst[Len():]. It fails as Score does, and when dst is shorter than
// Len(); after an error dst[:Len()] may be partly written and means
// nothing. It does not change query, retains neither query nor dst, and
// dst must not overlap query's or the batch's storage. A batch does not
// change once built: concurrent ScoreAll calls are safe, each with its own
// dst.
type EmbeddingBatch interface {
	Len() int
	Dims() int
	At(i int) Embedding
	ScoreAll(query Embedding, metric Metric, dst []float32) error
}

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
// score 0, and so do vectors so small that their float32 squared norms
// underflow (coordinates about 1e-23 or below), with growing error on the
// way there; Dense's cosine sums such vectors in float64 instead.
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
