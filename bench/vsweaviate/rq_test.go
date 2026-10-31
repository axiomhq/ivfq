package vsweaviate

// ivfq/rabitq against weaviate's 1-bit rotational quantizer
// (compressionhelpers.BinaryRotationalQuantizer): rotation cost, encode cost,
// query bind cost, scan cost per row, and estimator quality on one corpus.

import (
	"fmt"
	"math/rand/v2"
	"sort"
	"testing"

	"github.com/axiomhq/ivfq"
	"github.com/axiomhq/ivfq/internal/simd"
	"github.com/axiomhq/ivfq/rabitq"
	"github.com/weaviate/weaviate/adapters/repos/db/vector/compressionhelpers"
	"github.com/weaviate/weaviate/adapters/repos/db/vector/hnsw/distancer"
)

const rqRows = 1024 // rows per cluster: ivfq's default cluster size

func BenchmarkRotate(b *testing.B) {
	r := rand.New(rand.NewPCG(1, 2))
	for _, d := range dimsList {
		x := randF32(pool, d, r)
		ours := rabitq.NewRotation(rabitq.Seed("bench"), d)
		theirs := compressionhelpers.NewFastRotation(d, 3, compressionhelpers.DefaultFastRotationSeed)
		b.Run(fmt.Sprintf("d=%d/ivfq_Apply", d), func(b *testing.B) {
			v := make([]float32, d)
			pairs(b, 4*d, func(i int) { copy(v, x[i]); ours.Apply(v) })
			sinkF = v[0]
		})
		b.Run(fmt.Sprintf("d=%d/wv_signs+fwht_x3", d), func(b *testing.B) {
			out := make([]float32, theirs.OutputDim)
			pairs(b, 4*d, func(i int) { theirs.RotateInto(x[i], out) })
			sinkF = out[0]
		})
	}
}

// Encode: per-vector cost of making a 1-bit code. ivfq encodes a whole
// cluster (centroid, residuals, batched AVX2 rotation, bits, 2 scalars);
// weaviate encodes one vector at a time (rotate, sign bits, norm, step).
func BenchmarkEncode(b *testing.B) {
	r := rand.New(rand.NewPCG(1, 2))
	for _, d := range dimsList {
		vecs := randF32(rqRows, d, r)
		opts := rabitq.Options{Metric: ivfq.L2, Rotation: rabitq.NewRotation(rabitq.Seed("bench"), d)}
		b.Run(fmt.Sprintf("d=%d/ivfq_Quantize(1024)", d), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				if _, err := rabitq.Quantize(vecs, opts); err != nil {
					b.Fatal(err)
				}
			}
			b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N*rqRows), "ns/vec")
		})
		rq, err := compressionhelpers.NewBinaryRotationalQuantizer(d, compressionhelpers.DefaultFastRotationSeed, distancer.NewL2SquaredProvider())
		if err != nil {
			b.Fatal(err)
		}
		b.Run(fmt.Sprintf("d=%d/wv_Encode", d), func(b *testing.B) {
			pairs(b, 4*d, func(i int) { rq.Encode(vecs[i&(rqRows-1)]) })
		})
	}
}

// Bind: per-query, per-cluster setup (ivfq quantizes the query residual to the
// cluster centroid; weaviate quantizes the query once, to five planes, and
// reuses it for every code).
func BenchmarkBind(b *testing.B) {
	r := rand.New(rand.NewPCG(1, 2))
	for _, d := range dimsList {
		vecs := randF32(rqRows, d, r)
		qs := randF32(pool, d, r)
		rot := rabitq.NewRotation(rabitq.Seed("bench"), d)
		c, err := rabitq.Quantize(vecs, rabitq.Options{Metric: ivfq.L2, Rotation: rot})
		if err != nil {
			b.Fatal(err)
		}
		b.Run(fmt.Sprintf("d=%d/ivfq_NewQuery+Scorer", d), func(b *testing.B) {
			pairs(b, 4*d, func(i int) { _ = c.Scorer(rabitq.NewQuery(qs[i], ivfq.L2, rot)) })
		})
		// Global: the query is rotated and quantized once, then each column
		// binds in O(1). Global is the per-query cost, BindGlobal the per-column.
		b.Run(fmt.Sprintf("d=%d/ivfq_Global(per query)", d), func(b *testing.B) {
			pairs(b, 4*d, func(i int) { _ = rabitq.NewQuery(qs[i], ivfq.L2, rot).Global(rot) })
		})
		pc := c
		pc.PackFastScanFrame(rabitq.RotateCentroid(pc.Code.Centroid, rot))
		gq := rabitq.NewQuery(qs[0], ivfq.L2, rot).Global(rot)
		b.Run(fmt.Sprintf("d=%d/ivfq_BindGlobal(per column)", d), func(b *testing.B) {
			var s rabitq.Scorer
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				if !s.BindGlobal(&pc, gq) {
					b.Fatal("not bound")
				}
			}
		})
		rq, _ := compressionhelpers.NewBinaryRotationalQuantizer(d, compressionhelpers.DefaultFastRotationSeed, distancer.NewL2SquaredProvider())
		b.Run(fmt.Sprintf("d=%d/wv_NewDistancer", d), func(b *testing.B) {
			pairs(b, 4*d, func(i int) { _ = rq.NewDistancer(qs[i]) })
		})
	}
}

// Scan: score one 1024-row cluster; ns/row. ivfq has three paths: row at a
// time (Score), batched ScoreAll on the unpacked column, and ScoreAll on the
// PackFastScan layout (the one the engine runs). Weaviate: Distance per code.
func BenchmarkScan(b *testing.B) {
	r := rand.New(rand.NewPCG(1, 2))
	for _, d := range dimsList {
		vecs := randF32(rqRows, d, r)
		q := randF32(1, d, r)[0]
		rot := rabitq.NewRotation(rabitq.Seed("bench"), d)
		c, err := rabitq.Quantize(vecs, rabitq.Options{Metric: ivfq.L2, Rotation: rot})
		if err != nil {
			b.Fatal(err)
		}
		scores := make([]float32, rqRows)
		perRow := func(b *testing.B) {
			b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N*rqRows), "ns/row")
		}
		b.Run(fmt.Sprintf("d=%d/ivfq_Score(row)", d), func(b *testing.B) {
			s := c.Scorer(rabitq.NewQuery(q, ivfq.L2, rot))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				for row := 0; row < rqRows; row++ {
					scores[row] = s.Score(row)
				}
			}
			perRow(b)
		})
		b.Run(fmt.Sprintf("d=%d/ivfq_ScoreAll", d), func(b *testing.B) {
			s := c.Scorer(rabitq.NewQuery(q, ivfq.L2, rot))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				s.ScoreAll(scores, nil)
			}
			perRow(b)
		})
		b.Run(fmt.Sprintf("d=%d/ivfq_ScoreAll(packed)", d), func(b *testing.B) {
			p := c
			p.PackFastScan()
			s := p.Scorer(rabitq.NewQuery(q, ivfq.L2, rot))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				s.ScoreAll(scores, nil)
			}
			perRow(b)
		})
		b.Run(fmt.Sprintf("d=%d/ivfq_ScoreAll(packed,global)", d), func(b *testing.B) {
			p := c
			p.PackFastScanFrame(rabitq.RotateCentroid(p.Code.Centroid, rot))
			var s rabitq.Scorer
			if !s.BindGlobal(&p, rabitq.NewQuery(q, ivfq.L2, rot).Global(rot)) {
				b.Fatal("not bound")
			}
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				s.ScoreAll(scores, nil)
			}
			perRow(b)
		})
		rq, _ := compressionhelpers.NewBinaryRotationalQuantizer(d, compressionhelpers.DefaultFastRotationSeed, distancer.NewL2SquaredProvider())
		codes := make([][]uint64, rqRows)
		for i := range codes {
			codes[i] = rq.Encode(vecs[i])
		}
		b.Run(fmt.Sprintf("d=%d/wv_Distance", d), func(b *testing.B) {
			dq := rq.NewDistancer(q)
			var s float32
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				for _, code := range codes {
					v, _ := dq.Distance(code)
					s += v
				}
			}
			sinkF = s
			perRow(b)
		})
	}
}

// corpora: iid gaussian (hardest for any 1-bit code: no structure) and a
// gaussian mixture, 64 clusters, noise 0.3 of the centre spread.
func rqCorpus(kind string, n, d int, r *rand.Rand) (vecs [][]float32, label []int) {
	if kind == "gauss" {
		return randF32(n, d, r), make([]int, n)
	}
	const k = 64
	centres := randF32(k, d, r)
	vecs, label = make([][]float32, n), make([]int, n)
	for i := range vecs {
		label[i] = r.IntN(k)
		vecs[i] = make([]float32, d)
		for j := range vecs[i] {
			vecs[i][j] = centres[label[i]][j] + 0.3*float32(r.NormFloat64())
		}
	}
	return vecs, label
}

func topK(n, k int, less func(i, j int) bool) []int {
	ids := make([]int, n)
	for i := range ids {
		ids[i] = i
	}
	sort.Slice(ids, func(a, b int) bool { return less(ids[a], ids[b]) })
	return ids[:k]
}

func overlap(a, b []int) float64 {
	m := map[int]bool{}
	for _, x := range a {
		m[x] = true
	}
	var n int
	for _, x := range b {
		if m[x] {
			n++
		}
	}
	return float64(n) / float64(len(b))
}

// TestEstimatorQuality logs, per corpus: recall@10 of the top 10 by 1-bit
// estimate alone, and recall@10 inside the top 100 by estimate (what an exact
// rerank of 100 candidates would see). Run with -v. Lower-is-better distances.
func TestEstimatorQuality(t *testing.T) {
	if testing.Short() {
		t.Skip()
	}
	const n, nq = 8192, 50
	t.Logf("%-8s %5s  %-22s %9s %9s", "corpus", "dims", "codec", "R@10", "R@10/c100")
	for _, kind := range []string{"gauss", "mixture"} {
		for _, d := range []int{128, 384, 768} {
			r := rand.New(rand.NewPCG(7, uint64(d)))
			vecs, label := rqCorpus(kind, n, d, r)
			qv, _ := rqCorpus(kind, nq, d, rand.New(rand.NewPCG(8, uint64(d))))
			if kind == "mixture" { // queries from the same centres as the data
				r2 := rand.New(rand.NewPCG(7, uint64(d)))
				centres := randF32(64, d, r2)
				for i := range qv {
					centre := centres[r.IntN(64)]
					for j := range qv[i] {
						qv[i][j] = centre[j] + 0.3*float32(r.NormFloat64())
					}
				}
			}
			type codec struct {
				name  string
				score func(q []float32) []float32 // lower is better
			}
			rot := rabitq.NewRotation(rabitq.Seed("q"), d)
			one, err := rabitq.Quantize(vecs, rabitq.Options{Metric: ivfq.L2, Rotation: rot})
			if err != nil {
				t.Fatal(err)
			}
			one.PackFastScan()
			// ivfq as it runs: one column per cluster, centroid-relative.
			groups := map[int][]int{}
			for i, l := range label {
				groups[l] = append(groups[l], i)
			}
			cols := map[int]rabitq.Quantizer{}
			for l, ids := range groups {
				g := make([][]float32, len(ids))
				for j, id := range ids {
					g[j] = vecs[id]
				}
				c, err := rabitq.Quantize(g, rabitq.Options{Metric: ivfq.L2, Rotation: rot})
				if err != nil {
					t.Fatal(err)
				}
				cols[l] = c
			}
			rq, _ := compressionhelpers.NewBinaryRotationalQuantizer(d, compressionhelpers.DefaultFastRotationSeed, distancer.NewL2SquaredProvider())
			codes := make([][]uint64, n)
			for i := range codes {
				codes[i] = rq.Encode(vecs[i])
			}
			codecs := []codec{
				{"ivfq 1 column", func(q []float32) []float32 {
					s := make([]float32, n)
					c := one
					c.Scorer(rabitq.NewQuery(q, ivfq.L2, rot)).ScoreAll(s, nil)
					for i := range s {
						s[i] = -s[i]
					}
					return s
				}},
				{"wv RQ 1-bit", func(q []float32) []float32 {
					dq := rq.NewDistancer(q)
					s := make([]float32, n)
					for i, c := range codes {
						s[i], _ = dq.Distance(c)
					}
					return s
				}},
			}
			if kind == "mixture" {
				codecs = append(codecs[:1], append([]codec{{"ivfq per cluster", func(q []float32) []float32 {
					s := make([]float32, n)
					for l, c := range cols {
						sc := make([]float32, c.Rows())
						cc := c
						cc.Scorer(rabitq.NewQuery(q, ivfq.L2, rot)).ScoreAll(sc, nil)
						for j, id := range groups[l] {
							s[id] = -sc[j]
						}
					}
					return s
				}}}, codecs[1:]...)...)
			}
			for _, cd := range codecs {
				var r10, r100 float64
				for _, q := range qv {
					exact := topK(n, 10, func(i, j int) bool { return simd.L2Sq(q, vecs[i]) < simd.L2Sq(q, vecs[j]) })
					s := cd.score(q)
					less := func(i, j int) bool { return s[i] < s[j] }
					r10 += overlap(topK(n, 10, less), exact)
					r100 += overlap(topK(n, 100, less), exact)
				}
				t.Logf("%-8s %5d  %-22s %9.3f %9.3f", kind, d, cd.name, r10/nq, r100/nq)
			}
		}
	}
}
