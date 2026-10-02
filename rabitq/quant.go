package rabitq

import (
	"errors"
	"fmt"
	"github.com/axiomhq/ivfq"
	"math"
	"slices"
)

// Quantizer is one cluster's candidate column: one Code row per vector,
// dims bits plus 8 bytes each, encoded against the cluster's own centroid
// in the frame of one Rotation. Its bound is probabilistic (boundSigmas);
// Query.Exact trades it for certainty.
type Quantizer struct {
	Dims int
	// Code is the payload. Non-nil for anything UnmarshalBinary returned.
	Code *Code
}

// Rows is how many code rows the column holds, one per vector.
func (c Quantizer) Rows() int { return c.Code.Rows() }

// Retained is the bytes the column holds in memory of its own: zero for a
// column UnmarshalBinaryBorrowed aliased onto its input, the payload's size
// otherwise.
func (c Quantizer) Retained() int { return c.Code.retained() }

// Options is what Quantize needs beyond the vectors: the metric (cosine
// columns encode unit vectors) and the rotation, which also fixes the
// column's width so an empty cluster still publishes a shaped column.
type Options struct {
	Metric   ivfq.Metric
	Rotation *Rotation
}

// Quantize is the ONE door a hood's codes are written through. vectors are the hood's rows in id order, already at
// the precision the exact rerank will read them at.
func Quantize(vectors [][]float32, opts Options) (Quantizer, error) {
	if opts.Rotation == nil {
		return Quantizer{}, errors.New("rabitq: 1-bit codes need a rotation")
	}
	return quantizeBits(vectors, opts.Metric, opts.Rotation)
}

// Empty returns a column with no rows around centroid, in opts' frame:
// what AppendRows grows one block at a time, so every block of a cluster
// is encoded against the cluster's own centroid rather than the mean of
// whichever rows happened to start the block.
func Empty(centroid []float32, opts Options) (Quantizer, error) {
	if opts.Rotation == nil {
		return Quantizer{}, errors.New("rabitq: 1-bit codes need a rotation")
	}
	if opts.Rotation.dims <= 0 || len(centroid) != opts.Rotation.dims {
		return Quantizer{}, fmt.Errorf("rabitq: empty column needs a %d-wide centroid, got %d", opts.Rotation.dims, len(centroid))
	}
	if opts.Rotation.seed == 0 {
		return Quantizer{}, errors.New("rabitq: 1-bit codes need a nonzero rotation seed")
	}
	if opts.Metric != ivfq.L2 && opts.Metric != ivfq.Cosine {
		return Quantizer{}, fmt.Errorf("rabitq: metric %s has no 1-bit codec", opts.Metric)
	}
	unit := opts.Metric == ivfq.Cosine
	c := append([]float32(nil), centroid...)
	if unit {
		c = workRow(c, true)
	}
	return Quantizer{Dims: opts.Rotation.dims, Code: &Code{Centroid: c, Seed: opts.Rotation.seed, Unit: unit}}, nil
}

// Select returns a column holding only the rows at the given indexes, in
// that order, sharing the source column's centroid, seed and metric frame.
// Every per-row scalar and bit is copied verbatim, so a copied row decodes,
// scores and bounds exactly as it did in the source column.
//
// It is how a split writes each half's rows out of the source column
// instead of quantizing them again. A column is self-consistent around the
// ONE centroid its rows are residuals to, so rows may only be combined with
// rows of a column carrying that same centroid; AppendRows encodes new rows
// into it.
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
// It is how rows that join a column from elsewhere are encoded against
// this column's centroid, so the whole column stays one frame without
// re-quantizing the rows Select copied verbatim.
func AppendRows(c Quantizer, rot *Rotation, vectors [][]float32) (Quantizer, error) {
	b := c.Code
	if b == nil {
		return Quantizer{}, errors.New("rabitq: append on a column with no payload")
	}
	dims := c.Dims
	if rot == nil || rot.seed != b.Seed || rot.dims != dims {
		return Quantizer{}, fmt.Errorf("rabitq: append needs the column's rotation (seed %d, %d wide)", b.Seed, dims)
	}
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
	return c.Code.marshal(c.Dims, true)
}

// MarshalRows is MarshalBinary without the centroid: 4*Dims bytes fewer,
// for a store that files many code blocks under one cluster and keeps its
// centroid once. UnmarshalRowsBorrowed takes the centroid back.
func (c Quantizer) MarshalRows() ([]byte, error) {
	if c.Code == nil {
		return nil, fmt.Errorf("quant: codes with no payload")
	}
	return c.Code.marshal(c.Dims, false)
}

// UnmarshalRowsBorrowed decodes MarshalRows output against centroid, the
// one the rows were encoded against, validating it like
// UnmarshalBinaryBorrowed and keeping the rows in data. The caller must
// keep data and centroid immutable and alive; codes that share a centroid
// slice are one frame (SameFrame) without comparing it.
func UnmarshalRowsBorrowed(data []byte, centroid []float32) (Quantizer, error) {
	return decodeBits(data, false, true, centroid)
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
	// Rotation is the frame the columns were encoded in. A column whose
	// seed or width disagrees with it cannot be scored and bounds nothing.
	Rotation *Rotation
	// Exact drops the bound: every row's bound is +Inf, so the bound-pruned
	// pass reads every row and the answer is exactly the top k over the
	// probed clusters. It is how a caller buys back the certainty the
	// probabilistic bound trades away.
	Exact bool
	// Sigmas is the bound's width in standard deviations of the estimator's
	// error; 0 is the default, boundSigmas.
	Sigmas  float64
	invNorm float64 // 0 for a zero cosine query, which scores 0 everywhere
	// rotated is the query (unit, for cosine) in Rotation's frame, set by
	// Rotated: ScorerRotated takes a column's residual as rotated minus the
	// column's rotated centroid.
	rotated []float32
}

func NewQuery(q []float32, metric ivfq.Metric, rot *Rotation) Query {
	out := Query{Vector: q, Metric: metric, Rotation: rot}
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
func (c *Quantizer) Scorer(q Query) Scorer { return Scorer{bit: c.newBitScorer(q, nil)} }

// Rotated is q bound to rot with the query rotated once: a rotation is
// linear, so a column's rotated residual R(q-c) is Rq - Rc, and a scorer
// bound through ScorerRotated with the column's RotateCentroid does
// O(dims) work per column where Scorer rotates the residual of each (a
// lookup that probes hundreds of small clusters rotated its query
// hundreds of times).
func (q Query) Rotated(rot *Rotation) Query {
	q.Rotation = rot
	if rot == nil || len(q.Vector) < rot.dims {
		q.rotated = nil
		return q
	}
	v := make([]float32, rot.dims)
	copy(v, q.Vector[:rot.dims])
	if q.Metric == ivfq.Cosine && q.invNorm != 0 {
		inv := float32(q.invNorm)
		for j := range v {
			v[j] *= inv
		}
	}
	rot.Apply(v)
	q.rotated = v
	return q
}

// RotateCentroid is c in rot's frame, what ScorerRotated takes for a
// column encoded against c: compute it once per cluster, not per query.
func RotateCentroid(c []float32, rot *Rotation) []float32 {
	v := slices.Clone(c)
	rot.Apply(v)
	return v
}

// ScorerRotated is Scorer for a query from Rotated and the column's
// centroid in the same frame (RotateCentroid), its residual Rq - Rc. It
// scores as Scorer does, up to float rounding of the residual. A query
// not from Rotated, or rc of another width, binds as Scorer.
func (c *Quantizer) ScorerRotated(q Query, rc []float32) Scorer {
	if q.rotated == nil || len(rc) != c.Dims {
		rc = nil
	}
	return Scorer{bit: c.newBitScorer(q, rc)}
}

// Rebind is Scorer for a column in the same frame as the one s was built
// on (same centroid, rotation and metric): the query-side work is reused
// and only the rows change: a cluster stored as many blocks encoded against
// one centroid binds the query once per cluster rather than once per
// block. A column in another frame is bound afresh.
func (s Scorer) Rebind(c *Quantizer, q Query) Scorer {
	if s.bit != nil && s.bit.b.sameFrame(c.Code) {
		return s.With(c)
	}
	return c.Scorer(q)
}

// With is Rebind without the frame check, for a caller that has verified
// the columns share one frame (SameFrame).
func (s Scorer) With(c *Quantizer) Scorer {
	bit := *s.bit
	bit.b = c.Code
	return Scorer{bit: &bit}
}

// BindRotated is s = c.ScorerRotated(q, rc), reusing s's buffers: a
// worker that binds one hood after another allocates nothing once warm.
// Scorers copied from s (With) share its buffers and are invalidated.
func (s *Scorer) BindRotated(c *Quantizer, q Query, rc []float32) {
	if q.rotated == nil || len(rc) != c.Dims {
		rc = nil
	}
	if s.bit == nil {
		s.bit = new(bitScorer)
	}
	c.bind(s.bit, q, rc)
}

// Use is s = s.With(c) in place, without the copy.
func (s *Scorer) Use(c *Quantizer) { s.bit.b = c.Code }

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

// ScoreAll sets scores[i] to Score(i) for every row of the column, and
// bounds[i] to ScoreAndBound's bound when bounds is not nil, bit for bit:
// through the batched scan when the column is packed (PackFastScan), a
// row at a time otherwise. Both have at least Rows() entries.
func (s Scorer) ScoreAll(scores, bounds []float32) {
	scores = scores[:s.bit.b.Rows()]
	if bounds != nil {
		bounds = bounds[:len(scores)]
	}
	s.bit.scoreAll(scores, bounds)
}

// Score is ScoreAndBound's estimate alone, bit for bit, without the
// bound's arithmetic: a probe reads bounds only for its nearest hoods.
func (s Scorer) Score(row int) float32 { return s.bit.score(row) }

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
