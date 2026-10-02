package rabitq

import (
	"fmt"
	"math"
	"testing"

	"github.com/axiomhq/ivfq"
)

// ScoreAll is Score and ScoreAndBound row by row, bit for bit, packed or
// not, on typed and borrowed columns: widths that are not a multiple of
// four or 64, row counts either side of a 32-row block, both metrics, a
// zero row, a query that is a row, a zero query, and an exact query.
func TestScoreAllMatchesRowByRow(t *testing.T) {
	for _, metric := range []ivfq.Metric{ivfq.L2, ivfq.Cosine} {
		for _, dims := range []int{7, 64, 100, 128, 130} {
			for _, rows := range []int{0, 1, 31, 32, 33, 300} {
				v := corpus(rows, dims)
				if rows > 5 {
					v[5] = make([]float32, dims)
				}
				c, err := Quantize(v, Options{Metric: metric, Rotation: NewRotation(9, dims)})
				if err != nil {
					t.Fatal(err)
				}
				raw, err := c.MarshalBinary()
				if err != nil {
					t.Fatal(err)
				}
				borrowed, err := UnmarshalBinaryBorrowed(raw)
				if err != nil {
					t.Fatal(err)
				}
				queries := [][]float32{corpus(1, dims)[0], make([]float32, dims)}
				if rows > 3 {
					queries = append(queries, v[3])
				}
				for _, col := range []struct {
					name string
					q    Quantizer
				}{{"typed", c}, {"borrowed", borrowed}} {
					for _, packed := range []bool{false, true} {
						if packed {
							col.q.PackFastScan()
						}
						for qi, q := range queries {
							for _, exact := range []bool{false, true} {
								name := fmt.Sprintf("%v/d%d/r%d/%s/packed=%v/q%d/exact=%v", metric, dims, rows, col.name, packed, qi, exact)
								pq := NewQuery(q, metric, rotOf(col.q))
								pq.Exact = exact
								checkScoreAll(t, name, col.q.Scorer(pq), rows)
							}
						}
					}
				}
			}
		}
	}
}

func checkScoreAll(t *testing.T, name string, s Scorer, rows int) {
	t.Helper()
	scores, bounds := make([]float32, rows), make([]float32, rows)
	s.ScoreAll(scores, bounds)
	only := make([]float32, rows)
	s.ScoreAll(only, nil)
	for r := range rows {
		ws, wb := s.ScoreAndBound(r)
		if math.Float32bits(scores[r]) != math.Float32bits(ws) || math.Float32bits(bounds[r]) != math.Float32bits(wb) {
			t.Fatalf("%s row %d: ScoreAll %v/%v, ScoreAndBound %v/%v", name, r, scores[r], bounds[r], ws, wb)
		}
		if math.Float32bits(only[r]) != math.Float32bits(ws) {
			t.Fatalf("%s row %d: ScoreAll without bounds %v, ScoreAndBound %v", name, r, only[r], ws)
		}
	}
}

// A scorer rebound (With) from a packed column to an unpacked one of the
// same frame and back scores each by its own rows: the tables are the
// query's, the layout the column's.
func TestScoreAllAfterWith(t *testing.T) {
	const dims = 128
	v := corpus(100, dims)
	c, err := Quantize(v, Options{Metric: ivfq.L2, Rotation: NewRotation(3, dims)})
	if err != nil {
		t.Fatal(err)
	}
	raw, err := c.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	other, err := UnmarshalBinaryBorrowed(raw)
	if err != nil {
		t.Fatal(err)
	}
	c.PackFastScan()
	s := c.Scorer(NewQuery(v[10], ivfq.L2, rotOf(c)))
	checkScoreAll(t, "packed", s, 100)
	checkScoreAll(t, "with unpacked", s.With(&other), 100)
	checkScoreAll(t, "with packed again", s.With(&other).With(&c), 100)
}

func BenchmarkScoreAll(b *testing.B) {
	v := corpus(256, 128)
	c, err := Quantize(v, Options{Metric: ivfq.L2, Rotation: NewRotation(11, 128)})
	if err != nil {
		b.Fatal(err)
	}
	s := c.Scorer(NewQuery(v[0], ivfq.L2, rotOf(c)))
	scores, bounds := make([]float32, 256), make([]float32, 256)
	b.Run("rows", func(b *testing.B) {
		for b.Loop() {
			for r := range 256 {
				scores[r] = s.Score(r)
			}
		}
	})
	c.PackFastScan()
	b.Run("packed", func(b *testing.B) {
		for b.Loop() {
			s.ScoreAll(scores, nil)
		}
	})
	b.Run("packed+bounds", func(b *testing.B) {
		for b.Loop() {
			s.ScoreAll(scores, bounds)
		}
	})
}
