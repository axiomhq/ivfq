package vector_test

import (
	"context"
	"math"
	"math/rand"
	"slices"
	"sort"
	"testing"

	"github.com/axiomhq/ivfq"
	"github.com/axiomhq/ivfq/kmeans"
	"github.com/axiomhq/ivfq/rabitq"
	"github.com/axiomhq/ivfq/vector"
	"github.com/x448/float16"
)

// half decodes IEEE binary16 bits without the float16 package, so the
// expected values do not come from the decoder under test. Finite values only.
func half(bits uint16) float64 {
	sign := 1.0
	if bits>>15 == 1 {
		sign = -1
	}
	exp, frac := int(bits>>10)&0x1f, float64(bits&0x3ff)
	if exp == 0 {
		return sign * math.Ldexp(frac, -24)
	}
	return sign * math.Ldexp(1024+frac, exp-25)
}

func halves(bits ...uint16) []float16.Float16 {
	out := make([]float16.Float16, len(bits))
	for i, b := range bits {
		out[i] = float16.Frombits(b)
	}
	return out
}

func TestFloat32Rows(t *testing.T) {
	f32 := vector.Slice[float32]{Dims: 2, Data: []float32{0.1, -2.5, 3e38, -1e-38}}
	cases := []struct {
		name string
		rows func() ([][]float32, error)
		want [][]float32
	}{
		{"float32", f32.Float32Rows, [][]float32{{0.1, -2.5}, {3e38, -1e-38}}},
		// Read as integers, these bits are 15872, 50432, 1, 31743, 13653, 48128.
		{"float16", vector.Slice[float16.Float16]{Dims: 3, Data: halves(0x3E00, 0xC500, 0x0001, 0x7BFF, 0x3555, 0xBC00)}.Float32Rows,
			[][]float32{{1.5, -5, 5.9604645e-08}, {65504, 0.33325195, -1}}},
		{"int8", vector.Slice[int8]{Dims: 2, Data: []int8{-128, 127, -1, 0, 5, -7}}.Float32Rows,
			[][]float32{{-128, 127}, {-1, 0}, {5, -7}}},
		{"uint8", vector.Slice[uint8]{Dims: 3, Data: []uint8{255, 128, 0, 127, 200, 1}}.Float32Rows,
			[][]float32{{255, 128, 0}, {127, 200, 1}}},
		{"empty", vector.Slice[uint8]{Dims: 4}.Float32Rows, [][]float32{}},
	}
	for _, c := range cases {
		got, err := c.rows()
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		if len(got) != len(c.want) {
			t.Fatalf("%s: %d rows, want %d", c.name, len(got), len(c.want))
		}
		for i := range got {
			if !slices.Equal(got[i], c.want[i]) {
				t.Fatalf("%s: row %d = %v, want %v", c.name, i, got[i], c.want[i])
			}
		}
		if len(got) > 1 {
			next := got[1][0]
			_ = append(got[0], 42)
			if got[1][0] != next {
				t.Fatalf("%s: appending to row 0 overwrote row 1", c.name)
			}
		}
	}

	rows, _ := f32.Float32Rows()
	if &rows[0][0] != &f32.Data[0] || &rows[1][0] != &f32.Data[2] {
		t.Fatal("float32 rows copied their coordinates")
	}
	if !slices.Equal(f32.Data, []float32{0.1, -2.5, 3e38, -1e-38}) {
		t.Fatalf("source changed: %v", f32.Data)
	}
}

func TestFloat32RowsRejects(t *testing.T) {
	nan, inf := float32(math.NaN()), float32(math.Inf(1))
	for name, rows := range map[string]func() ([][]float32, error){
		"zero dims":       vector.Slice[float32]{Dims: 0, Data: []float32{1, 2}}.Float32Rows,
		"zero dims empty": vector.Slice[int8]{}.Float32Rows,
		"negative dims":   vector.Slice[uint8]{Dims: -2, Data: []uint8{1, 2}}.Float32Rows,
		"partial row":     vector.Slice[int8]{Dims: 2, Data: []int8{1, 2, 3}}.Float32Rows,
		"float32 NaN":     vector.Slice[float32]{Dims: 2, Data: []float32{1, 2, 3, nan}}.Float32Rows,
		"float32 +Inf":    vector.Slice[float32]{Dims: 2, Data: []float32{1, inf, 3, 4}}.Float32Rows,
		"float32 -Inf":    vector.Slice[float32]{Dims: 1, Data: []float32{1, -inf}}.Float32Rows,
		"float16 qNaN":    vector.Slice[float16.Float16]{Dims: 2, Data: halves(0x3C00, 0x7E00)}.Float32Rows,
		"float16 sNaN":    vector.Slice[float16.Float16]{Dims: 2, Data: halves(0x3C00, 0x7C01)}.Float32Rows,
		"float16 -NaN":    vector.Slice[float16.Float16]{Dims: 2, Data: halves(0x3C00, 0xFE00)}.Float32Rows,
		"float16 +Inf":    vector.Slice[float16.Float16]{Dims: 1, Data: halves(0x3C00, 0x7C00)}.Float32Rows,
		"float16 -Inf":    vector.Slice[float16.Float16]{Dims: 1, Data: halves(0xFC00)}.Float32Rows,
	} {
		if got, err := rows(); err == nil {
			t.Errorf("%s: accepted as %v", name, got)
		}
	}
}

// TestPipeline runs every scalar type through the real IVFQ pipeline (fit,
// split, centroids, routing, RaBitQ quantize and append, bound-pruned
// scoring with exact rerank) and checks each step against brute force over
// the stored values, decoded independently. Blob 0 is two close sub-blobs,
// so a 4-way fit keeps them together and the split separates them.
func TestPipeline(t *testing.T) {
	const dims, perBlob, sub = 16, 40, 5
	rng := rand.New(rand.NewSource(7))
	centers := make([][]float64, sub)
	for b := range centers {
		centers[b] = make([]float64, dims)
		for d := range centers[b] {
			centers[b][d] = rng.Float64()*1.6 - 0.8
			if b == 1 {
				centers[1][d] = centers[0][d] + 0.15
			}
		}
	}
	var ideal [][]float64
	var subLabel []int
	for b, c := range centers {
		for range perBlob {
			v := make([]float64, dims)
			for d := range v {
				v[d] = max(-1, min(1, c[d]+rng.NormFloat64()*0.02))
			}
			ideal = append(ideal, v)
			subLabel = append(subLabel, b)
		}
	}
	t.Run("float32", func(t *testing.T) {
		pipeline(t, encode(ideal, dims, func(x float64) (float32, float64) { return float32(x), float64(float32(x)) }), subLabel)
	})
	t.Run("float16", func(t *testing.T) {
		pipeline(t, encode(ideal, dims, func(x float64) (float16.Float16, float64) {
			h := float16.Fromfloat32(float32(1.5 + 3*x))
			return h, half(h.Bits())
		}), subLabel)
	})
	t.Run("int8", func(t *testing.T) {
		pipeline(t, encode(ideal, dims, func(x float64) (int8, float64) {
			v := int8(math.Round(120 * x))
			return v, float64(v)
		}), subLabel)
	})
	t.Run("uint8", func(t *testing.T) {
		pipeline(t, encode(ideal, dims, func(x float64) (uint8, float64) {
			v := uint8(math.Round(128 + 120*x))
			return v, float64(v)
		}), subLabel)
	})
}

type stored[T vector.Scalar] struct {
	s     vector.Slice[T]
	truth [][]float64 // the stored values, decoded by the test
}

func encode[T vector.Scalar](ideal [][]float64, dims int, enc func(float64) (T, float64)) stored[T] {
	out := stored[T]{s: vector.Slice[T]{Dims: dims}}
	for _, row := range ideal {
		tr := make([]float64, dims)
		for d, x := range row {
			var v T
			v, tr[d] = enc(x)
			out.s.Data = append(out.s.Data, v)
		}
		out.truth = append(out.truth, tr)
	}
	return out
}

func pipeline[T vector.Scalar](t *testing.T, in stored[T], subLabel []int) {
	ctx := context.Background()
	dims, truth := in.s.Dims, in.truth
	source := slices.Clone(in.s.Data)
	rows, err := in.s.Float32Rows()
	if err != nil {
		t.Fatal(err)
	}
	for i := range truth {
		for d := range truth[i] {
			if float64(rows[i][d]) != truth[i][d] {
				t.Fatalf("row %d coordinate %d = %v, stored %v", i, d, rows[i][d], truth[i][d])
			}
		}
	}
	snapshot := make([][]float32, len(rows))
	for i := range rows {
		snapshot[i] = slices.Clone(rows[i])
	}

	// Fit: blob 0 (sub-blobs 0 and 1) stays one cluster.
	top := make([]int, len(subLabel))
	for i, l := range subLabel {
		top[i] = max(0, l-1)
	}
	cents, assign, err := kmeans.Config{K: 4, Iters: 20, Seed: 1}.Fit(ctx, rows)
	if err != nil {
		t.Fatal(err)
	}
	if !samePartition(assign, top) {
		t.Fatalf("fit partition %v, want %v", assign, top)
	}
	for i, a := range assign {
		if want := order(truth[i], cents, l2)[0]; a != want {
			t.Fatalf("row %d assigned to %d, nearest centroid is %d", i, a, want)
		}
	}
	members := groups(assign, len(cents))
	for k, ids := range members {
		mean := mean(truth, ids)
		c, _ := ivfq.Centroid(pick(rows, ids), dims, false)
		near(t, "ivfq.Centroid", c, mean)
		near(t, "kmeans centroid", cents[k], mean)
	}

	// Split blob 0's cluster in two, as a split does.
	parent := assign[0]
	ids := members[parent]
	halfCents, sides, err := kmeans.Config{K: 2, Iters: kmeans.SplitIters, Seed: 2}.FitBalanced(ctx, pick(rows, ids))
	if err != nil {
		t.Fatal(err)
	}
	var subs []int
	for _, id := range ids {
		subs = append(subs, subLabel[id])
	}
	if !samePartition(sides, subs) {
		t.Fatalf("split partition %v, want %v", sides, subs)
	}
	split := groups(sides, 2)
	for j := range split {
		if len(split[j]) < (len(ids)+3)/4 {
			t.Fatalf("split side %d holds %d of %d", j, len(split[j]), len(ids))
		}
		var global []int
		for _, s := range split[j] {
			global = append(global, ids[s])
		}
		near(t, "split centroid", halfCents[j], mean(truth, global))
		split[j] = global
	}
	cents = append(cents, halfCents[1])
	cents[parent] = halfCents[0]
	members = append(members, split[1])
	members[parent] = split[0]
	cluster := make([]int, len(rows))
	for k, ids := range members {
		for _, id := range ids {
			cluster[id] = k
		}
	}

	// Routing: the tree ranks the centroids as brute force does, and sends
	// every stored row to its own cluster.
	tree, err := ivfq.Build(ctx, cents, 2)
	if err != nil {
		t.Fatal(err)
	}
	var queries [][]float32
	for i := 0; i+37 < len(rows); i += 13 {
		q := make([]float32, dims)
		for d := range q {
			q[d] = float32((truth[i][d] + truth[i+37][d]) / 2)
		}
		queries = append(queries, q)
	}
	for _, q := range queries {
		if got, want := tree.Nearest(q, len(cents)), order(f64(q), cents, l2); !slices.Equal(got, want) {
			t.Fatalf("tree routes %v as %v, brute force %v", q, got, want)
		}
	}
	for i, r := range rows {
		if got := tree.Nearest(r, 1)[0]; got != cluster[i] {
			t.Fatalf("row %d routed to %d, it is in %d", i, got, cluster[i])
		}
	}

	// RaBitQ: each cluster quantizes half its rows and appends the rest;
	// a query scores every routed cluster, reranks a shortlist exactly and
	// reads every other row whose bound reaches the shortlist's k-th exact
	// score. With Query.Exact that is brute force's top k; the default bound
	// is probabilistic, so it is held to a recall floor.
	const k = 5
	rot := rabitq.NewRotation(rabitq.Seed("vector", "pipeline"), dims)
	for _, metric := range []ivfq.Metric{ivfq.L2, ivfq.Cosine} {
		opts := rabitq.Options{Metric: metric, Rotation: rot}
		cols := make([]rabitq.Quantizer, len(members))
		for c, ids := range members {
			h := len(ids) / 2
			col, err := rabitq.Quantize(pick(rows, ids[:h]), opts)
			if err != nil {
				t.Fatal(err)
			}
			if cols[c], err = rabitq.AppendRows(col, rot, pick(rows, ids[h:])); err != nil {
				t.Fatal(err)
			}
			if cols[c].Rows() != len(ids) {
				t.Fatalf("cluster %d has %d code rows, want %d", c, cols[c].Rows(), len(ids))
			}
		}
		for _, exactQuery := range []bool{false, true} {
			var read, found int
			for _, q := range queries {
				type cand struct {
					id         int
					est, bound float32
				}
				var cands []cand
				for _, c := range tree.Nearest(q, len(cents)) {
					n := len(members[c])
					scores, bounds := make([]float32, n), make([]float32, n)
					query := rabitq.NewQuery(q, metric, rot)
					query.Exact = exactQuery
					cols[c].Scorer(query).ScoreAll(scores, bounds)
					for r, id := range members[c] {
						cands = append(cands, cand{id, scores[r], bounds[r]})
					}
				}
				exact := func(id int) float32 { return ivfq.Score(metric, q, rows[id]) }
				sort.SliceStable(cands, func(a, b int) bool { return cands[a].est > cands[b].est })
				var shortlist []int
				for _, c := range cands[:2*k] {
					shortlist = append(shortlist, c.id)
				}
				cutoff := exact(rerank(shortlist, exact)[k-1])
				for _, c := range cands[2*k:] {
					if c.bound >= cutoff {
						shortlist = append(shortlist, c.id)
					}
				}
				read += len(shortlist)
				got := rerank(shortlist, exact)[:k]
				want := bruteTop(truth, f64(q), metric, k)
				for i := range got {
					if slices.Contains(want, got[i]) {
						found++
					}
					if g, w := score(metric, f64(q), truth[got[i]]), score(metric, f64(q), truth[want[i]]); exactQuery && math.Abs(g-w) > 1e-5*math.Abs(w)+1e-9 {
						t.Fatalf("%s: hit %d is row %d scoring %v, brute force has row %d scoring %v", metric, i, got[i], g, want[i], w)
					}
				}
			}
			recall := float64(found) / float64(k*len(queries))
			t.Logf("%s exact=%v: recall@%d %.3f, read %d of %d rows per query", metric, exactQuery, k, recall, read/len(queries), len(rows))
			if !exactQuery && recall < 0.9 {
				t.Fatalf("%s: recall@%d %.3f with the default bound", metric, k, recall)
			}
		}
	}

	if !slices.Equal(in.s.Data, source) {
		t.Fatal("the pipeline changed the source data")
	}
	for i := range rows {
		if !slices.Equal(rows[i], snapshot[i]) {
			t.Fatalf("the pipeline changed row %d", i)
		}
	}
}

func l2(a []float64, b []float32) float64 {
	var s float64
	for i := range a {
		d := a[i] - float64(b[i])
		s += d * d
	}
	return s
}

// order ranks cents by distance to v, nearest first.
func order(v []float64, cents [][]float32, dist func([]float64, []float32) float64) []int {
	ids := make([]int, len(cents))
	for i := range ids {
		ids[i] = i
	}
	sort.SliceStable(ids, func(a, b int) bool { return dist(v, cents[ids[a]]) < dist(v, cents[ids[b]]) })
	return ids
}

func score(metric ivfq.Metric, q, v []float64) float64 {
	var dot, qq, vv, l2 float64
	for i := range q {
		dot, qq, vv = dot+q[i]*v[i], qq+q[i]*q[i], vv+v[i]*v[i]
		l2 += (q[i] - v[i]) * (q[i] - v[i])
	}
	if metric == ivfq.Cosine {
		return dot / math.Sqrt(qq*vv)
	}
	return -l2
}

func bruteTop(truth [][]float64, q []float64, metric ivfq.Metric, k int) []int {
	ids := make([]int, len(truth))
	for i := range ids {
		ids[i] = i
	}
	sort.SliceStable(ids, func(a, b int) bool { return score(metric, q, truth[ids[a]]) > score(metric, q, truth[ids[b]]) })
	return ids[:k]
}

func rerank(ids []int, exact func(int) float32) []int {
	ids = slices.Clone(ids)
	sort.SliceStable(ids, func(a, b int) bool { return exact(ids[a]) > exact(ids[b]) })
	return ids
}

func samePartition(a, b []int) bool {
	ab, ba := map[int]int{}, map[int]int{}
	for i := range a {
		if x, ok := ab[a[i]]; ok && x != b[i] {
			return false
		}
		if y, ok := ba[b[i]]; ok && y != a[i] {
			return false
		}
		ab[a[i]], ba[b[i]] = b[i], a[i]
	}
	return len(a) == len(b)
}

func groups(assign []int, k int) [][]int {
	out := make([][]int, k)
	for i, a := range assign {
		out[a] = append(out[a], i)
	}
	return out
}

func pick(rows [][]float32, ids []int) [][]float32 {
	out := make([][]float32, len(ids))
	for i, id := range ids {
		out[i] = rows[id]
	}
	return out
}

func mean(truth [][]float64, ids []int) []float64 {
	m := make([]float64, len(truth[0]))
	for _, id := range ids {
		for d, x := range truth[id] {
			m[d] += x / float64(len(ids))
		}
	}
	return m
}

func f64(v []float32) []float64 {
	out := make([]float64, len(v))
	for i, x := range v {
		out[i] = float64(x)
	}
	return out
}

func near(t *testing.T, what string, got []float32, want []float64) {
	t.Helper()
	for d := range want {
		if math.Abs(float64(got[d])-want[d]) > 1e-4*(1+math.Abs(want[d])) {
			t.Fatalf("%s coordinate %d = %v, brute-force mean %v", what, d, got[d], want[d])
		}
	}
}
