package rabitq

import (
	"bytes"
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

// A scorer rebound with BindRotated scores as a fresh ScorerRotated, bit
// for bit, column after column: across widths, after a bind that could
// not score (wrong metric), and after Use moved it to another column.
func TestBindRotatedReusesNothingStale(t *testing.T) {
	var s Scorer
	for i, dims := range []int{128, 100, 7, 128} {
		rot := NewRotation(uint64(20+i), dims)
		v := corpus(70, dims)
		c, err := Quantize(v, Options{Metric: ivfq.L2, Rotation: rot})
		if err != nil {
			t.Fatal(err)
		}
		c.PackFastScan()
		cent := make([]float32, dims)
		for j := range cent {
			cent[j] = c.Code.centroid(j)
		}
		rc := RotateCentroid(cent, rot)
		for qi, q := range [][]float32{v[1], cent, v[2]} {
			if qi == 1 {
				// A cosine query against l2 codes cannot score: a dead
				// bind between two live ones. The others rebind a live one.
				s.BindRotated(&c, NewQuery(q, ivfq.Cosine, rot).Rotated(rot), rc)
			}
			pq := NewQuery(q, ivfq.L2, rot).Rotated(rot)
			s.BindRotated(&c, pq, rc)
			fresh := c.ScorerRotated(pq, rc)
			name := fmt.Sprintf("dims %d query %d", dims, qi)
			want, wantB := make([]float32, 70), make([]float32, 70)
			fresh.ScoreAll(want, wantB)
			got, gotB := make([]float32, 70), make([]float32, 70)
			s.ScoreAll(got, gotB)
			for r := range want {
				if math.Float32bits(got[r]) != math.Float32bits(want[r]) || math.Float32bits(gotB[r]) != math.Float32bits(wantB[r]) {
					t.Fatalf("%s row %d: rebound %v/%v, fresh %v/%v", name, r, got[r], gotB[r], want[r], wantB[r])
				}
			}
			checkScoreAll(t, name, s, 70)
		}
	}
	// Use moves the binding to a column of the same frame.
	rot := NewRotation(5, 64)
	v := corpus(40, 64)
	c, err := Quantize(v, Options{Metric: ivfq.L2, Rotation: rot})
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
	other.PackFastScan()
	s.BindRotated(&c, NewQuery(v[2], ivfq.L2, rot), nil)
	s.Use(&other)
	checkScoreAll(t, "use", s, 40)
}

// Packing replaces a column's bits: every score, bound, the marshalled
// bytes, a Select of its rows and its frame are what they were before,
// on typed and borrowed columns, widths not a multiple of 4 or 64, row
// counts either side of a block, both metrics and a zero row.
func TestPackedColumnIsTheColumn(t *testing.T) {
	for _, metric := range []ivfq.Metric{ivfq.L2, ivfq.Cosine} {
		for _, dims := range []int{7, 100, 128, 130} {
			for _, rows := range []int{1, 31, 33, 100} {
				v := corpus(rows, dims)
				if rows > 5 {
					v[5] = make([]float32, dims)
				}
				rot := NewRotation(13, dims)
				c, err := Quantize(v, Options{Metric: metric, Rotation: rot})
				if err != nil {
					t.Fatal(err)
				}
				raw, err := c.MarshalBinary()
				if err != nil {
					t.Fatal(err)
				}
				for _, kind := range []string{"typed", "borrowed"} {
					name := fmt.Sprintf("%v/d%d/r%d/%s", metric, dims, rows, kind)
					col, err := UnmarshalBinary(raw)
					if kind == "borrowed" {
						col, err = UnmarshalBinaryBorrowed(bytes.Clone(raw))
					}
					if err != nil {
						t.Fatal(err)
					}
					ref, err := UnmarshalBinary(raw) // never packed
					if err != nil {
						t.Fatal(err)
					}
					pq := NewQuery(corpus(1, dims)[0], metric, rot)
					wantS, wantB := make([]float32, rows), make([]float32, rows)
					ref.Scorer(pq).ScoreAll(wantS, wantB)
					col.PackFastScan()
					s := col.Scorer(pq)
					gotS, gotB := make([]float32, rows), make([]float32, rows)
					s.ScoreAll(gotS, gotB)
					for r := range rows {
						rs, rb := s.ScoreAndBound(r)
						for _, pair := range [][2]float32{{gotS[r], wantS[r]}, {gotB[r], wantB[r]}, {rs, wantS[r]}, {rb, wantB[r]}} {
							if math.Float32bits(pair[0]) != math.Float32bits(pair[1]) {
								t.Fatalf("%s row %d: packed %v/%v (row path %v/%v), unpacked %v/%v", name, r, gotS[r], gotB[r], rs, rb, wantS[r], wantB[r])
							}
						}
					}
					again, err := col.MarshalBinary()
					if err != nil || !bytes.Equal(again, raw) {
						t.Fatalf("%s: packed column marshals differently (%v)", name, err)
					}
					sel := []int{rows - 1, 0}
					a, errA := col.Select(sel)
					b, errB := ref.Select(sel)
					if errA != nil || errB != nil {
						t.Fatal(errA, errB)
					}
					ab, _ := a.MarshalBinary()
					bb, _ := b.MarshalBinary()
					if !bytes.Equal(ab, bb) {
						t.Fatalf("%s: Select of the packed column differs", name)
					}
					if !col.SameFrame(&ref) || !ref.SameFrame(&col) {
						t.Fatalf("%s: packing changed the column's frame", name)
					}
				}
			}
		}
	}
}

// A packed column charges what it holds: a centroid only when it owns
// one (copied out of borrowed bytes), not the cluster's centroid that
// every block decoded with UnmarshalRowsBorrowed shares; and with a shared
// centroid, less than the same rows unpacked.
func TestPackedRetained(t *testing.T) {
	const dims, rows = 128, 64
	v := corpus(rows, dims)
	c, err := Quantize(v, Options{Metric: ivfq.L2, Rotation: NewRotation(3, dims)})
	if err != nil {
		t.Fatal(err)
	}
	rowsBytes, err := c.MarshalRows()
	if err != nil {
		t.Fatal(err)
	}
	shared, err := UnmarshalRowsBorrowed(rowsBytes, c.Code.Centroid)
	if err != nil {
		t.Fatal(err)
	}
	shared.PackFastScan()
	raw, err := c.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	owned, err := UnmarshalBinaryBorrowed(raw)
	if err != nil {
		t.Fatal(err)
	}
	owned.PackFastScan()
	if d := owned.Retained() - shared.Retained(); d != 4*dims {
		t.Fatalf("owned centroid charges %d more than a shared one, want %d", d, 4*dims)
	}
	if shared.Retained() >= c.Retained() {
		t.Fatalf("packed with a shared centroid charges %d, the unpacked column %d", shared.Retained(), c.Retained())
	}
}
