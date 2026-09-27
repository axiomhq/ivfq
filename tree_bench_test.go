package ivfq

import (
	"bufio"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"math/rand"
	"os"
	"runtime"
	"sync"
	"testing"
	"time"
)

// The three benchmarks share one 10,000 × 128 tree, the shape the other
// tree benchmarks measure: Build is the cost an upsert-heavy caller
// defers, Clone is what it pays to mutate a cached tree safely, and an
// Upsert append is the incremental path those two exist to avoid.

func BenchmarkTreeBuild(b *testing.B) {
	v := treeData(10000, 128)
	b.ReportAllocs()
	for b.Loop() {
		if _, err := Build(context.Background(), v, 100); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkTreeUpsertAppend(b *testing.B) {
	v := treeData(10000, 128)
	base := mustBuild(v, 100)
	tree := base.Clone()
	pool := make([][]float32, 1024)
	r := rand.New(rand.NewSource(5))
	for i := range pool {
		nv := make([]float32, 128)
		for j := range nv {
			nv[j] = float32(r.NormFloat64())
		}
		pool[i] = nv
	}
	// Appends grow holders without bound, so every burst of 1,024 restarts
	// from a fresh clone: steady-state append cost, with the rare clone
	// amortized into ~0.1% of the measured operations.
	const burst = 1024
	done := 0
	b.ReportAllocs()
	for b.Loop() {
		if done == burst {
			tree = base.Clone()
			done = 0
		}
		if err := tree.Upsert(tree.Leaves(), pool[done]); err != nil {
			b.Fatal(err)
		}
		done++
	}
}

func BenchmarkTreeClone(b *testing.B) {
	v := treeData(10000, 128)
	base := mustBuild(v, 100)
	b.ReportAllocs()
	for b.Loop() {
		_ = base.Clone()
	}
}

// The beam-accuracy benchmark answers one question with real data: how far
// can the assignment beam drop before the tree stops returning the flat
// nearest centroid? The beam of 16 was picked on a 1,000-centroid tree one
// level deep, where a beam step already covers the whole frontier; at the
// 10M build's k = 9,766 the tree is two levels over a ~98-wide root fan, so
// the beam's cost is beam × ~100 leaf evaluations and its recall is whatever
// the top-beam root children happen to contain. It streams the first rows of
// the BIGANN base set (u8bin: two little-endian uint32, rows and dims, then
// row-major uint8; the same layout cmd/bench streams), trains centroids the
// way a fold does — RunSampledBudget at kmeansIters over a fixed-seed
// sample — and measures, for every beam, ns per row of a GOMAXPROCS=8
// assignment pass, agreement against a brute-force nearest scan, the mean
// ratio of the assigned centroid's squared distance to the exact nearest's,
// and how often the assigner misses entirely.
//
// The corpus path is beamCorpusEnv-overridable; without the file the
// benchmark skips, and -short skips it too. Fixtures are cached per k, so
// -count re-runs re-measure without retraining.

const beamCorpusEnv = "IVF_BEAM_CORPUS"

const beamCorpus = "" // set IVF_BEAM_CORPUS to a BIGANN .u8bin file

// beamSeed fixes every random draw: sample selection and k-means seeding.
const beamSeed = int64(0x6bea4)

// beamIters is the Lloyd cap an index build trains with; the benchmark must
// measure the same centroids a build would grow.
const beamIters = 10

// beamFanout is the tree fan an index build uses.
const beamFanout = 100

type beamFixture struct {
	centroids [][]float32
	tree      Tree
	queries   [][]float32
	exactID   []int
	exactD    []float32
}

// beamTrain is the fanout-independent part of a fixture: the trained
// centroids, the held-out rows, and the exact scan over them. Building a
// tree at another fanout reuses all of it.
type beamTrain struct {
	centroids [][]float32
	queries   [][]float32
	exactID   []int
	exactD    []float32
}

type beamKey struct{ k, fanout int }

var (
	beamFixtureMu   sync.Mutex
	beamTrainByK    = map[int]*beamTrain{}
	beamFixtureByKF = map[beamKey]*beamFixture{}
)

// beamStream reads u8bin rows in batches, converting each row to float32 the
// way cmd/bench's reader does.
type beamStream struct {
	f    *os.File
	r    *bufio.Reader
	dims int
	row  int
	rows int
	buf  []byte
}

func openBeamStream(b *testing.B, path string) *beamStream {
	f, err := os.Open(path)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			b.Skipf("corpus %s unavailable (set %s to point elsewhere)", path, beamCorpusEnv)
		}
		b.Fatal(err)
	}
	var h [2]uint32
	if err := binary.Read(f, binary.LittleEndian, &h); err != nil {
		f.Close()
		b.Fatal(err)
	}
	if h[0] == 0 || h[1] == 0 {
		f.Close()
		b.Fatalf("%s: empty corpus", path)
	}
	return &beamStream{
		f:    f,
		r:    bufio.NewReaderSize(f, 8<<20),
		dims: int(h[1]),
		rows: int(h[0]),
		buf:  make([]byte, h[1]),
	}
}

func (s *beamStream) close() { s.f.Close() }

// next returns up to n rows as float32 vectors, io.EOF once the file ends.
func (s *beamStream) next(n int) ([][]float32, error) {
	out := make([][]float32, 0, n)
	for len(out) < n && s.row < s.rows {
		if _, err := io.ReadFull(s.r, s.buf); err != nil {
			return out, fmt.Errorf("%s: row %d: %w", s.f.Name(), s.row, err)
		}
		v := make([]float32, s.dims)
		for i, x := range s.buf {
			v[i] = float32(x)
		}
		out = append(out, v)
		s.row++
	}
	if len(out) == 0 {
		return nil, io.EOF
	}
	return out, nil
}

// loadBeamFixture trains k centroids over the corpus and prepares the
// held-out measurement rows, then builds the tree at the given fanout.
// Two training shapes, mirroring the two tables the beam decision needs:
//
//	sample: train over a 32×k fixed-seed sample of the first 1,000,000
//	        rows (SampleSize's ratio), measure on the first 100,000 rows
//	        outside the sample.
//	full:   train over all of the first 1,000,000 rows, measure on rows
//	        1,000,000 through 1,100,000, which training never saw.
func loadBeamFixture(b *testing.B, k int, sample bool, fanout int) *beamFixture {
	beamFixtureMu.Lock()
	defer beamFixtureMu.Unlock()
	key := beamKey{k, fanout}
	if fx, ok := beamFixtureByKF[key]; ok {
		return fx
	}
	train, ok := beamTrainByK[k]
	if !ok {
		train = loadBeamTrain(b, k, sample)
		beamTrainByK[k] = train
	}
	tree, err := Build(context.Background(), train.centroids, fanout)
	if err != nil {
		b.Fatal(err)
	}
	b.Logf("k=%d fanout=%d: tree height %d over %d leaves", k, fanout, tree.Height(), tree.Leaves())
	logTreeShape(b, &tree, train.queries)
	fx := &beamFixture{
		centroids: train.centroids,
		tree:      tree,
		queries:   train.queries,
		exactID:   train.exactID,
		exactD:    train.exactD,
	}
	beamFixtureByKF[key] = fx
	return fx
}

// loadBeamTrain streams the corpus once and trains k centroids the way a
// build does. The draw and the fit are pure functions of the seed.
func loadBeamTrain(b *testing.B, k int, sample bool) *beamTrain {
	path := os.Getenv(beamCorpusEnv)
	if path == "" {
		path = beamCorpus
	}
	s := openBeamStream(b, path)
	defer s.close()

	const poolRows = 1_000_000
	const queryRows = 100_000
	if s.rows < poolRows+queryRows {
		b.Fatalf("%s: %d rows, need %d", path, s.rows, poolRows+queryRows)
	}
	// In-sample marks the training rows; everything else streams to the
	// query set in input order, so the draw is a pure function of the seed.
	inSample := make([]bool, poolRows)
	trainN := poolRows
	if sample {
		trainN = 32 * k
		rng := rand.New(rand.NewSource(beamSeed))
		for _, j := range rng.Perm(poolRows)[:trainN] {
			inSample[j] = true
		}
	}
	train := make([][]float32, 0, trainN)
	queries := make([][]float32, 0, queryRows)
	for len(train) < trainN || len(queries) < queryRows {
		rows, err := s.next(4096)
		if err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			b.Fatal(err)
		}
		base := s.row - len(rows)
		for bi, v := range rows {
			idx := base + bi
			// Sample mode holds out every unsampled pool row for
			// measurement; rows past the pool are measurement-only by
			// construction. Everything else trains.
			if idx < poolRows && sample && !inSample[idx] {
				if len(queries) < queryRows {
					queries = append(queries, v)
				}
				continue
			}
			if idx >= poolRows {
				if len(queries) < queryRows {
					queries = append(queries, v)
				}
				continue
			}
			train = append(train, v)
		}
	}
	if len(train) < trainN || len(queries) < queryRows {
		b.Fatalf("%s: streamed only %d training and %d query rows, need %d and %d", path, len(train), len(queries), trainN, queryRows)
	}

	b.Logf("k=%d: training %d rows, measuring %d held-out rows, dims %d", k, len(train), len(queries), s.dims)
	centroids, _, err := RunSampledBudget(context.Background(), train, k, beamIters, beamSeed, 0)
	if err != nil {
		b.Fatal(err)
	}
	exactID, exactD := beamExact(centroids, queries)
	return &beamTrain{
		centroids: centroids,
		queries:   queries,
		exactID:   exactID,
		exactD:    exactD,
	}
}

// logTreeShape records the shape a routing assignment descends: the root
// fan, how unevenly the first level splits the leaves, and the evaluation
// counts a beam-16 and a top-4 two-stage assignment pay per row.
func logTreeShape(b *testing.B, tree *Tree, qs [][]float32) {
	root := &tree.nodes[tree.root]
	if len(root.children) == 0 {
		return
	}
	leavesUnder := func(id int) int {
		n := &tree.nodes[id]
		if len(n.children) == 0 {
			return len(n.leaves)
		}
		total := 0
		for _, h := range n.sub {
			total += len(tree.nodes[h].leaves)
		}
		return total
	}
	lo, hi, sum := 1<<30, 0, 0
	for _, c := range root.children {
		n := leavesUnder(c)
		lo, hi, sum = min(lo, n), max(hi, n), sum+n
	}
	fan := len(root.children)
	b.Logf("root fan %d, leaves per subtree min/avg/max %d/%.0f/%d", fan, lo, float64(sum)/float64(fan), hi)
	const probe = 1_000
	meanEvals := func(search func([]float32, int) ([]int, int), x int) int {
		total := 0
		for i := 0; i < probe; i++ {
			_, e := search(qs[i%len(qs)], x)
			total += e
		}
		return total / probe
	}
	b.Logf("mean evals/row over %d rows: beam 16 = %d, two-stage top 4 = %d",
		probe, meanEvals(tree.Evaluations, 16), meanEvals(tree.TwoStageNearest, 4))
}

// beamExact brute-forces the nearest centroid per row — ivf.Nearest's rule,
// strictly smaller wins and the first index holds ties — computing the
// distance once so the ratio metric can reuse it.
func beamExact(centroids, qs [][]float32) ([]int, []float32) {
	ids := make([]int, len(qs))
	ds := make([]float32, len(qs))
	var wg sync.WaitGroup
	workers := min(runtime.GOMAXPROCS(0), len(qs))
	chunk := (len(qs) + workers - 1) / workers
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for i := w * chunk; i < min((w+1)*chunk, len(qs)); i++ {
				v := qs[i]
				best := 0
				bestD := L2Sq(centroids[0], v)
				for j := 1; j < len(centroids); j++ {
					if d := L2Sq(centroids[j], v); d < bestD {
						best, bestD = j, d
					}
				}
				ids[i], ds[i] = best, bestD
			}
		}(w)
	}
	wg.Wait()
	return ids, ds
}

// beamAssign runs one full assignment pass over the queries, fanned out over
// GOMAXPROCS workers the way the maintainer's route batches are.
func beamAssign(qs [][]float32, assign func([]float32) int) []int {
	ids := make([]int, len(qs))
	var wg sync.WaitGroup
	workers := min(runtime.GOMAXPROCS(0), len(qs))
	chunk := (len(qs) + workers - 1) / workers
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for i := w * chunk; i < min((w+1)*chunk, len(qs)); i++ {
				ids[i] = assign(qs[i])
			}
		}(w)
	}
	wg.Wait()
	return ids
}

// BenchmarkAssignerBeam measures Tree.Assigner's cost and accuracy per beam
// at the two centroid counts the routing decision sits between: k =
// HoodK(1,000,000) trained over a 32×k sample, and the 10M build's k = 9,766
// trained over the same 32×k sample. One operation is one parallel
// assignment pass over the 100,000 held-out rows; ns/row is what a level
// merge's cycle-1 routing pays per row. The fanout screen exists because a
// beam step costs beam × leaves-per-holder plus the internal levels: a
// smaller fan shrinks every holder scan and gives the level-one k-means's
// imbalance more headroom before the tree deepens, at a wider stage-one
// ranking that is still exact.
func BenchmarkAssignerBeam(b *testing.B) {
	if testing.Short() {
		b.Skip("corpus benchmark: skipped in -short")
	}
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(8))
	type beamSpec struct {
		k        int
		sample   bool
		fanout   int
		beams    []int
		twoStage bool
	}
	fullBeams := []int{1, 2, 4, 8, 16, 32}
	specs := []beamSpec{
		{k: HoodK(1_000_000), sample: true, fanout: beamFanout, beams: fullBeams, twoStage: true},
		// The 10M build's k, trained over the same 32×k sample a build
		// trains over (SampleSize, the shape bulkframe's frame trainer
		// uses), because that is the artifact a 10M cycle actually routes
		// against. Full-1M training was measured first and dropped: it
		// exceeded the shared-host budget by minutes without moving the
		// tree (height 5 either way over these centroids).
		{k: 9_766, sample: true, fanout: beamFanout, beams: fullBeams, twoStage: true},
		// The fanout screen at the 10M build's k, beam 16's accuracy bar
		// held fixed.
		{k: 9_766, sample: true, fanout: 25, beams: []int{8, 16, 32}},
		{k: 9_766, sample: true, fanout: 32, beams: []int{8, 16, 32}},
		{k: 9_766, sample: true, fanout: 50, beams: []int{8, 16, 32}},
	}
	for _, sp := range specs {
		b.Run(fmt.Sprintf("k=%d/f=%d", sp.k, sp.fanout), func(b *testing.B) {
			fx := loadBeamFixture(b, sp.k, sp.sample, sp.fanout)
			for _, beam := range sp.beams {
				assign := fx.tree.Assigner(beam)
				b.Run(fmt.Sprintf("beam=%d", beam), func(b *testing.B) {
					runAssignPass(b, fx, assign)
				})
			}
			if sp.twoStage {
				for _, top := range []int{1, 2, 4, 8} {
					assign := fx.tree.TwoStageAssigner(top)
					b.Run(fmt.Sprintf("2stage=%d", top), func(b *testing.B) {
						runAssignPass(b, fx, assign)
					})
				}
			}
		})
	}
}

// runAssignPass times one parallel assignment pass over the held-out rows
// and reports ns per row, exact agreement against the brute-force scan, the
// mean distance ratio, and the miss rate.
func runAssignPass(b *testing.B, fx *beamFixture, assign func([]float32) int) {
	rows := float64(len(fx.queries))
	b.ReportAllocs()
	b.ResetTimer()
	var wall time.Duration
	var ids []int
	for b.Loop() {
		start := time.Now()
		ids = beamAssign(fx.queries, assign)
		wall += time.Since(start)
	}
	b.StopTimer()
	agree, neg, ratioN := 0, 0, 0
	var ratioSum float64
	for i, id := range ids {
		if id < 0 {
			neg++
			continue
		}
		if id == fx.exactID[i] {
			agree++
		}
		if fx.exactD[i] > 0 {
			ratioSum += float64(L2Sq(fx.centroids[id], fx.queries[i])) / float64(fx.exactD[i])
			ratioN++
		}
	}
	b.ReportMetric(float64(wall.Nanoseconds())/float64(b.N)/rows, "ns/row")
	b.ReportMetric(float64(agree)/rows*100, "agree%")
	b.ReportMetric(ratioSum/float64(ratioN), "dist-ratio")
	b.ReportMetric(float64(neg)/rows*100, "neg%")
}
