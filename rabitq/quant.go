package rabitq

import (
	"fmt"
	"github.com/axiomhq/ivfq"
	"math"
)

// Codes is one hood's candidate column: the bytes the probe wave moves and
// the kernel scores. There is ONE codec — RaBitQ 1-bit residuals
// (rabitq.go), dims BITS plus 8 bytes per row, with a bound that holds at
// boundConfidence — and this type is the seam the engine sees.
//
// Until 2026-09-15 there were two, selected by Config.CandidateCodes, and a
// hood's payload said which one it held so mixed generations read. The
// per-vector symmetric int8 codec (DQC2) is gone: it cost dims BYTES plus 16
// a row against 1-bit's dims bits plus 8, which on SIFT1M was 105.3 stored
// bytes a row against 27.8, and the one thing it bought — a bound that
// cannot be exceeded — is bought back where it is actually wanted by
// Query.Exact.
type Quantizer struct {
	Dims int
	// Code is the payload. Non-nil for anything UnmarshalBinary returned.
	Code *Code
}

// Count is how many code rows this column holds — one per vector the
// manifest counted, which is what CheckParts holds it to.
func (c Quantizer) Rows() int { return c.Code.Rows() }

// Retained is what a decoded column holds in memory, for the byte cache's
// accounting.
func (c Quantizer) Retained() int { return c.Code.retained() }

// Options is what Encode needs beyond the vectors: the field's declared
// width (an empty hood still publishes a shaped column), the metric (cosine
// hoods encode unit vectors), and the rotation seed.
type Options struct {
	Dims   int
	Metric ivfq.Metric
	Seed   uint64
}

// Quantize is the ONE door a hood's codes are written through. vectors are the hood's rows in id order, already at
// the precision the exact rerank will read them at.
func Quantize(vectors [][]float32, opts Options) (Quantizer, error) {
	dims := opts.Dims
	if dims <= 0 && len(vectors) > 0 {
		dims = len(vectors[0])
	}
	return quantizeBits(vectors, dims, opts.Metric, opts.Seed)
}

// Empty returns a column with no rows around centroid, in opts' frame:
// what AppendRows grows one block at a time, so every block of a cluster
// is encoded against the cluster's own centroid rather than the mean of
// whichever rows happened to start the block.
func Empty(centroid []float32, opts Options) (Quantizer, error) {
	if opts.Dims <= 0 || len(centroid) != opts.Dims {
		return Quantizer{}, fmt.Errorf("quant: empty column needs a %d-wide centroid, got %d", opts.Dims, len(centroid))
	}
	if opts.Seed == 0 {
		return Quantizer{}, fmt.Errorf("quant: 1-bit codes need a nonzero rotation seed")
	}
	if opts.Metric != ivfq.L2 && opts.Metric != ivfq.Cosine {
		return Quantizer{}, fmt.Errorf("rabitq: metric %s has no 1-bit codec", opts.Metric)
	}
	unit := opts.Metric == ivfq.Cosine
	c := append([]float32(nil), centroid...)
	if unit {
		c = workRow(c, true)
	}
	return Quantizer{Dims: opts.Dims, Code: &Code{Centroid: c, Seed: opts.Seed, Unit: unit}}, nil
}

// Select returns a column holding only the rows at the given indexes, in
// that order, sharing the source column's centroid, seed and metric frame.
// Every per-row scalar and bit is copied verbatim, so a copied row decodes,
// scores and bounds exactly as it did in the source column.
//
// This is the read-and-reassemble half of copying codes between packs: a
// split writes each half's rows out of the source hood's column instead of
// quantizing them again. A column is
// self-consistent around the ONE centroid its rows are residuals to, so
// rows may only be combined with rows of a column carrying that same
// centroid — that is what AppendRows is for.
func (c Quantizer) Select(rows []int) (Quantizer, error) {
	b := c.Code
	if b == nil {
		return Quantizer{}, fmt.Errorf("quant: select on a column with no payload")
	}
	w := bitWords(c.Dims)
	out := &Code{
		Words:    make([]uint64, len(rows)*w),
		Centroid: b.Centroid,
		Norms:    make([]float32, len(rows)),
		Aligns:   make([]float32, len(rows)),
		Seed:     b.Seed,
		Unit:     b.Unit,
	}
	if b.borrowed != nil {
		out.Centroid = make([]float32, c.Dims)
		for i := range out.Centroid {
			out.Centroid[i] = b.centroid(i)
		}
	}
	for j, r := range rows {
		if r < 0 || r >= b.Rows() {
			return Quantizer{}, fmt.Errorf("quant: row %d out of range for a %d-row column", r, b.Rows())
		}
		if b.borrowed != nil {
			for i := 0; i < w; i++ {
				out.Words[j*w+i] = b.word(r*w + i)
			}
			out.Norms[j], out.Aligns[j] = b.norm(r), b.align(r)
		} else {
			copy(out.Words[j*w:(j+1)*w], b.Words[r*w:(r+1)*w])
			out.Norms[j], out.Aligns[j] = b.Norms[r], b.Aligns[r]
		}
	}
	return Quantizer{Dims: c.Dims, Code: out}, nil
}

// AppendRows returns a copy of the column with one freshly encoded row per
// vector, appended in order, encoded against the column's OWN centroid,
// seed and metric frame. The zero-vector rule carries over: a zero vector
// appended to a cosine column becomes a zeroRowAlign row.
//
// It is the other half of copying codes between packs: the few rows that
// move INTO a split's new hood from a neighbouring one are re-encoded
// against the copied column's centroid, so the whole column stays one
// frame without re-quantizing the rows that were copied verbatim.
func AppendRows(c Quantizer, vectors [][]float32) (Quantizer, error) {
	b := c.Code
	if b == nil {
		return Quantizer{}, fmt.Errorf("quant: append on a column with no payload")
	}
	dims := c.Dims
	w := bitWords(dims)
	rows := b.Rows()
	extra := len(vectors)
	out := &Code{
		Words:    make([]uint64, (rows+extra)*w),
		Centroid: b.Centroid,
		Norms:    make([]float32, rows+extra),
		Aligns:   make([]float32, rows+extra),
		Seed:     b.Seed,
		Unit:     b.Unit,
	}
	if b.borrowed != nil {
		out.Centroid = make([]float32, dims)
		for i := range out.Centroid {
			out.Centroid[i] = b.centroid(i)
		}
	}
	if b.borrowed != nil {
		for i := 0; i < rows*w; i++ {
			out.Words[i] = b.word(i)
		}
		for i := 0; i < rows; i++ {
			out.Norms[i], out.Aligns[i] = b.norm(i), b.align(i)
		}
	} else {
		copy(out.Words, b.Words)
		copy(out.Norms, b.Norms)
		copy(out.Aligns, b.Aligns)
	}
	rot := rotationFor(b.Seed, dims)
	scratch := make([]float32, dims)
	for i, v := range vectors {
		if len(v) != dims {
			return Quantizer{}, fmt.Errorf("quant: appended row %d is %d-wide, the column is %d", i, len(v), dims)
		}
		row := workRow(v, b.Unit)
		if row == nil {
			out.Norms[rows+i], out.Aligns[rows+i] = 0, zeroRowAlign
			continue
		}
		out.fillRow(rows+i, row, rot, scratch)
	}
	return Quantizer{Dims: dims, Code: out}, nil
}

func (c Quantizer) MarshalBinary() ([]byte, error) {
	if c.Code == nil {
		return nil, fmt.Errorf("quant: codes with no payload")
	}
	return c.Code.marshal(c.Dims)
}

// UnmarshalBinary decodes MarshalBinary output. Everything a row's score
// and bound rest on is checked there (rabitq.go, unmarshalBits), because the
// candidate pass prunes the exact rerank with these values before any full
// vector is read.
func UnmarshalBinary(data []byte) (Quantizer, error) {
	if len(data) >= 4 && string(data[:4]) == bitsMagic {
		return unmarshalBits(data)
	}
	return Quantizer{}, fmt.Errorf("quant: codes are not %s", bitsMagic)
}

// UnmarshalBinaryBorrowed validates codes like UnmarshalBinary, but keeps
// scalar and bit rows in data. The caller must keep data immutable and alive.
func UnmarshalBinaryBorrowed(data []byte) (Quantizer, error) {
	if len(data) >= 4 && string(data[:4]) == bitsMagic {
		return unmarshalBitsBorrowed(data)
	}
	return Quantizer{}, fmt.Errorf("quant: codes are not %s", bitsMagic)
}

// Query is a vector prepared for scoring many rows: the cosine query norm is
// computed once here rather than per candidate.
type Query struct {
	Vector []float32
	Metric ivfq.Metric
	// Exact drops the bound: every row's bound is +Inf, so the bound-pruned
	// pass reads every row and the answer is exactly the top k over the
	// probed hoods. It is how a caller buys back the certainty the 1-bit
	// codec's probabilistic bound trades away.
	Exact bool
	// Sigmas overrides boundSigmas for the 1-bit codec (0 = the default).
	// Tests measure coverage against it; the query path sets it from
	// search.Config.BoundSigmas, which the bound-width sweep varies.
	Sigmas  float64
	invNorm float64 // 0 for a zero cosine query, which scores 0 everywhere
}

func NewQuery(q []float32, metric ivfq.Metric) Query {
	out := Query{Vector: q, Metric: metric}
	if metric == ivfq.Cosine {
		var qn float64
		for _, x := range q {
			qn += float64(x) * float64(x)
		}
		if qn != 0 {
			out.invNorm = 1 / math.Sqrt(qn)
		}
	}
	return out
}

// Scorer binds one column to one query: the per-hood work the paper's
// estimator needs — the query's residual to THIS hood's centroid, rotated
// and quantized to four bit planes — which is O(dims) once per probed hood
// rather than once per row. The candidate pass builds one per hood and
// scores its rows through it.
//
// A Scorer belongs to the goroutine that made it.
type Scorer struct {
	bit *bitScorer
}

// Scorer prepares q against this column.
func (c *Quantizer) Scorer(q Query) Scorer { return Scorer{bit: c.newBitScorer(q)} }

// Rebind is Scorer for a column in the same frame as the one s was built
// on (same centroid, rotation and metric): the query-side work is reused
// and only the rows change. A cluster of the LSM tier is many 64-row
// blocks encoded against one centroid, so a probe binds the query once per
// cluster rather than once per block. A column in another frame is bound
// afresh.
func (s Scorer) Rebind(c *Quantizer, q Query) Scorer {
	if s.bit != nil && s.bit.b.sameFrame(c.Code) {
		return s.With(c)
	}
	return c.Scorer(q)
}

// With is Rebind without the frame check, for a caller that has verified
// the columns share one frame (tier.ClusterView.SameFrame).
func (s Scorer) With(c *Quantizer) Scorer {
	bit := *s.bit
	bit.b = c.Code
	return Scorer{bit: &bit}
}

// SameFrame reports whether o's rows were encoded exactly as c's.
func (c *Quantizer) SameFrame(o *Quantizer) bool { return c.Code.sameFrame(o.Code) }

// ScoreAndBound is the candidate kernel: a higher-is-better estimate of the
// exact score, and a conservative upper bound on it. A row may be skipped by
// an exact top-k cutoff only when its bound is below the cutoff.
//
// The bound is boundSigmas standard deviations of the RaBitQ estimator's
// error and holds at boundConfidence, so "exact within the probed hoods" is
// a recall guarantee rather than a certainty — unless Query.Exact is set,
// which is how a caller buys the certainty back.
func (s Scorer) ScoreAndBound(row int) (score, bound float32) { return s.bit.scoreAndBound(row) }

// Score is ScoreAndBound's estimate alone.
func (s Scorer) Score(row int) float32 {
	score, _ := s.bit.scoreAndBound(row)
	return score
}

// Score is the one-off form of Scorer.Score: it binds the query for a single
// row. Tests and oracles use it; the candidate pass does not, because the
// binding is O(dims).
func (c Quantizer) Score(q Query, row int) float32 { return c.Scorer(q).Score(row) }

// UpperBound is the one-off form of ScoreAndBound's bound.
func (c Quantizer) UpperBound(q Query, row int) float32 {
	_, bound := c.Scorer(q).ScoreAndBound(row)
	return bound
}

// ScoreAndBound is the one-off form of Scorer.ScoreAndBound.
func (c Quantizer) ScoreAndBound(q Query, row int) (score, bound float32) {
	return c.Scorer(q).ScoreAndBound(row)
}

// finiteNonnegative is the check every stored scalar of a decoded column
// has to pass before the candidate pass prunes anything with it.
func finiteNonnegative(x float32) bool {
	return !math.IsNaN(float64(x)) && !math.IsInf(float64(x), 0) && x >= 0
}

type ranked struct {
	id    int
	score float32
}

// Codes keeps the name existing importers already use.
type Codes = Quantizer
