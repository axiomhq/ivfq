package rabitq

import (
	"github.com/axiomhq/ivfq"
	"testing"
)

// A scorer rebound to another column in the same frame (Scorer.With, the
// tier's per-cluster binding) scores that column's rows exactly as a
// scorer bound to it directly, whichever representation either column has
// and however many rows it holds.
func TestReboundScorerReadsTheNewColumn(t *testing.T) {
	for _, metric := range []ivfq.Metric{ivfq.L2, ivfq.Cosine} {
		opts := Options{Metric: metric, Rotation: NewRotation(11, 129)}
		centroid := corpus(1, 129)[0]
		small, err := Empty(centroid, opts)
		if err != nil {
			t.Fatal(err)
		}
		if small, err = AppendRows(small, opts.Rotation, corpus(5, 129)); err != nil {
			t.Fatal(err)
		}
		big, err := Empty(centroid, opts)
		if err != nil {
			t.Fatal(err)
		}
		if big, err = AppendRows(big, opts.Rotation, corpus(64, 129)[5:]); err != nil {
			t.Fatal(err)
		}
		data, err := big.MarshalBinary()
		if err != nil {
			t.Fatal(err)
		}
		bigBorrowed, err := UnmarshalBinaryBorrowed(data)
		if err != nil {
			t.Fatal(err)
		}
		q := NewQuery(corpus(2, 129)[1], metric, opts.Rotation)
		for _, target := range []Quantizer{big, bigBorrowed} {
			if !small.SameFrame(&target) {
				t.Fatalf("%s: columns are not in one frame", metric)
			}
			rebound := small.Scorer(q).With(&target)
			direct := target.Scorer(q)
			for row := 0; row < target.Rows(); row++ {
				gs, gb := rebound.ScoreAndBound(row)
				ws, wb := direct.ScoreAndBound(row)
				if gs != ws || gb != wb {
					t.Fatalf("%s row %d: rebound (%v,%v), direct (%v,%v)", metric, row, gs, gb, ws, wb)
				}
			}
		}
	}
}
