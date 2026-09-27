package rank

import "errors"

// rrfK is the standard reciprocal-rank-fusion damping constant: large enough
// that rank differences deep in a leg matter little, small enough that the
// top ranks dominate.
const rrfK = 60

// Fusion configures reciprocal rank fusion. The zero value is the standard
// RRF: rank constant 60, every leg weighted 1.
type Fusion struct {
	RankConstant int       `json:"rank_constant,omitempty"` // <= 0 means 60
	Weights      []float64 `json:"weights,omitempty"`       // one per leg, > 0; nil means all 1
}

// Validate checks f against the number of legs it will fuse.
func (f Fusion) Validate(legs int) error {
	if f.RankConstant < 0 {
		return errors.New("rank_constant must be positive")
	}
	if f.Weights != nil && len(f.Weights) != legs {
		return errors.New("weights must name one weight per query")
	}
	for _, w := range f.Weights {
		if !(w > 0) {
			return errors.New("weights must be positive")
		}
	}
	return nil
}

// RRF fuses per-leg rankings by reciprocal rank fusion: each doc scores
// Σ 1/(60+rank) over the legs that returned it (rank is 1-based position;
// a doc repeated within one leg counts its best rank once). Scores are
// rank-derived, not the legs' raw scores. Result is the fused top k, ties
// broken by id ascending so results are deterministic.
func RRF(k int, legs ...[]Hit) []Hit { return Fusion{}.RRF(k, legs...) }

// RRF is RRF with f's rank constant and per-leg weights:
// Σ weight/(rank_constant+rank). A leg without a weight counts as 1;
// Validate rejects that mismatch before it reaches here.
func (f Fusion) RRF(k int, legs ...[]Hit) []Hit {
	c := f.RankConstant
	if c <= 0 {
		c = rrfK
	}
	fused := map[string]float64{}
	for i, leg := range legs {
		w := 1.0
		if i < len(f.Weights) {
			w = f.Weights[i]
		}
		seen := map[string]bool{}
		for rank, h := range leg {
			if seen[h.ID] {
				continue
			}
			seen[h.ID] = true
			fused[h.ID] += w / float64(c+rank+1)
		}
	}
	if len(fused) == 0 {
		return nil
	}
	out := make([]Hit, 0, len(fused))
	for id, s := range fused {
		out = append(out, Hit{ID: id, Score: float32(s)})
	}
	return Select(out, k)
}
