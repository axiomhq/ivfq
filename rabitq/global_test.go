package rabitq

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/axiomhq/ivfq"
)

// globalColumn is v quantized and packed with its frame, for BindGlobal.
func globalColumn(t testing.TB, v [][]float32, metric ivfq.Metric) (Quantizer, *Rotation) {
	t.Helper()
	rot := NewRotation(17, len(v[0]))
	c, err := Quantize(v, Options{Metric: metric, Rotation: rot})
	if err != nil {
		t.Fatal(err)
	}
	cent := make([]float32, c.Dims)
	for i := range cent {
		cent[i] = c.Code.centroid(i)
	}
	c.PackFastScanFrame(RotateCentroid(cent, rot))
	return c, rot
}

// offCentre is a corpus far from the origin, as BIGANN's non-negative
// bytes are: a global query's rounding error is largest there relative to
// its distance to a column's centroid.
func offCentre(n, d int, seed int64) [][]float32 {
	r := rand.New(rand.NewSource(seed))
	out := make([][]float32, n)
	for i := range out {
		out[i] = make([]float32, d)
		for j := range out[i] {
			out[i][j] = 100 + float32(r.NormFloat64()*12)
		}
	}
	return out
}

// A global query's ScoreAll is its ScoreAndBound and Score row by row, bit
// for bit: widths not a multiple of 4 or 64, rows either side of a block,
// both metrics, a zero row, a query that is a row, the zero query, exact.
func TestGlobalScoreAllMatchesRowByRow(t *testing.T) {
	for _, metric := range []ivfq.Metric{ivfq.L2, ivfq.Cosine} {
		for _, dims := range []int{7, 100, 128, 130} {
			for _, rows := range []int{1, 33, 100} {
				v := corpus(rows, dims)
				if rows > 5 {
					v[5] = make([]float32, dims)
				}
				c, rot := globalColumn(t, v, metric)
				for qi, q := range [][]float32{corpus(1, dims)[0], v[0], make([]float32, dims)} {
					for _, exact := range []bool{false, true} {
						pq := NewQuery(q, metric, rot).Global(rot)
						pq.Exact = exact
						var s Scorer
						if !s.BindGlobal(&c, pq) {
							t.Fatalf("%v d%d: a packed column and a global query did not bind", metric, dims)
						}
						checkScoreAll(t, fmt.Sprintf("global %v/d%d/r%d/q%d/exact=%v", metric, dims, rows, qi, exact), s, rows)
					}
				}
			}
		}
	}
}

// BindGlobal declines what it cannot score: a query that is not global, a
// column packed without its frame, a column of another rotation.
func TestBindGlobalDeclines(t *testing.T) {
	v := corpus(50, 64)
	c, rot := globalColumn(t, v, ivfq.L2)
	var s Scorer
	if s.BindGlobal(&c, NewQuery(v[1], ivfq.L2, rot).Rotated(rot)) {
		t.Fatal("bound a query that is not global")
	}
	plain, err := Quantize(v, Options{Metric: ivfq.L2, Rotation: rot})
	if err != nil {
		t.Fatal(err)
	}
	plain.PackFastScan()
	if s.BindGlobal(&plain, NewQuery(v[1], ivfq.L2, rot).Global(rot)) {
		t.Fatal("bound a column packed without its frame")
	}
	other := NewRotation(99, 64)
	if s.BindGlobal(&c, NewQuery(v[1], ivfq.L2, other).Global(other)) {
		t.Fatal("bound a query of another rotation")
	}
}

// The global query's bound holds at its confidence, as the per-column
// bound does (TestBitBoundCoversAtItsConfidence), on the same corpora and
// one far from the origin. It also logs the estimate's RMS error and the
// bound's mean width beside the per-column scorer's.
func TestGlobalBoundCoversAtItsConfidence(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	axis := func(n, d int) [][]float32 {
		out := make([][]float32, n)
		for i := range out {
			out[i] = make([]float32, d)
			for j := 0; j < 6; j++ {
				out[i][rng.Intn(d)] = float32(rng.NormFloat64() * 4)
			}
		}
		return out
	}
	for _, tc := range []struct {
		name string
		v, q [][]float32
	}{
		{"gaussian", corpus(3000, 128), corpus(40, 128)},
		{"clustered", clusteredCorpus(3000, 16, 128), clusteredCorpus(40, 16, 128)},
		{"axis-aligned", axis(3000, 128), axis(40, 128)},
		{"off-centre", offCentre(3000, 128, 1), offCentre(40, 128, 2)},
		{"off-centre-768", offCentre(1000, 768, 3), offCentre(20, 768, 4)},
	} {
		for _, metric := range []ivfq.Metric{ivfq.L2, ivfq.Cosine} {
			c, rot := globalColumn(t, tc.v, metric)
			for _, sigmas := range []float64{2, boundSigmas} {
				var over, total int
				var errG, errC, widthG, widthC float64
				for _, q := range tc.q {
					pq := NewQuery(q, metric, rot).Global(rot)
					pq.Sigmas = sigmas
					var g Scorer
					if !g.BindGlobal(&c, pq) {
						t.Fatal("did not bind")
					}
					col := c.Scorer(func() Query { x := NewQuery(q, metric, rot); x.Sigmas = sigmas; return x }())
					for row := range tc.v {
						exact := float64(ivfq.Score(metric, q, tc.v[row]))
						sg, bg := g.ScoreAndBound(row)
						sc, bc := col.ScoreAndBound(row)
						if exact > float64(bg) {
							over++
						}
						total++
						errG += (float64(sg) - exact) * (float64(sg) - exact)
						errC += (float64(sc) - exact) * (float64(sc) - exact)
						widthG += float64(bg) - float64(sg)
						widthC += float64(bc) - float64(sc)
					}
				}
				n := float64(total)
				share := float64(over) / n
				tail := 0.5 * math.Erfc(sigmas/math.Sqrt2)
				t.Logf("%s %s %.1fσ: over %.5f (tail %.5f); rms error global %.4g per-column %.4g; bound width global %.4g per-column %.4g",
					tc.name, metric, sigmas, share, tail, math.Sqrt(errG/n), math.Sqrt(errC/n), widthG/n, widthC/n)
				if share > 4*tail {
					t.Fatalf("%s %s at %.1f sigmas: exceeded by %.5f of rows, past the %.5f it claims", tc.name, metric, sigmas, share, tail)
				}
			}
		}
	}
}
