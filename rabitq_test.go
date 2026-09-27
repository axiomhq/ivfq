package ivfq

import (
	"encoding/binary"
	"math"
	"math/rand"
	"reflect"
	"testing"
)

func bitOpts(metric string) Options {
	return Options{Metric: metric, Seed: Seed("ns", "vector")}
}

func mustEncode(t *testing.T, vectors [][]float32, opts Options) Quantizer {
	t.Helper()
	c, err := Quantize(vectors, opts)
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func TestBitCodesBinaryRoundTrip(t *testing.T) {
	for _, metric := range []string{"l2", "cosine"} {
		vectors := corpus(64, 33) // 33 dims: the last word is mostly padding
		vectors[7] = make([]float32, 33)
		want := mustEncode(t, vectors, bitOpts(metric))
		if want.Rows() != len(vectors) || want.Dims != 33 {
			t.Fatalf("%s: shape %d x %d", metric, want.Rows(), want.Dims)
		}
		b, err := want.MarshalBinary()
		if err != nil {
			t.Fatal(err)
		}
		if n := bitsHeader + 4*33 + 8*64 + 8*64*1; len(b) != n {
			t.Fatalf("%s: %d bytes, want %d", metric, len(b), n)
		}
		got, err := UnmarshalBinary(b)
		if err != nil || !reflect.DeepEqual(got, want) {
			t.Fatalf("%s round trip: %v", metric, err)
		}
		// An empty hood still publishes a shaped column.
		empty := mustEncode(t, nil, Options{Dims: 8, Metric: metric, Seed: 7})
		eb, err := empty.MarshalBinary()
		if err != nil {
			t.Fatal(err)
		}
		if back, err := UnmarshalBinary(eb); err != nil || back.Rows() != 0 || back.Dims != 8 {
			t.Fatalf("%s empty: %+v %v", metric, back, err)
		}
	}
}

func TestBorrowedBitCodesMatchOwned(t *testing.T) {
	for _, metric := range []string{"l2", "cosine"} {
		vectors := corpus(64, 129)
		data, err := mustEncode(t, vectors, bitOpts(metric)).MarshalBinary()
		if err != nil {
			t.Fatal(err)
		}
		owned, err := UnmarshalBinary(data)
		if err != nil {
			t.Fatal(err)
		}
		borrowed, err := UnmarshalBinaryBorrowed(data)
		if err != nil {
			t.Fatal(err)
		}
		if borrowed.Rows() != owned.Rows() || borrowed.Retained() != 0 {
			t.Fatalf("%s borrowed rows=%d retained=%d", metric, borrowed.Rows(), borrowed.Retained())
		}
		q := NewQuery(vectors[3], metric)
		for row := 0; row < borrowed.Rows(); row++ {
			gotScore, gotBound := borrowed.ScoreAndBound(q, row)
			wantScore, wantBound := owned.ScoreAndBound(q, row)
			if gotScore != wantScore || gotBound != wantBound {
				t.Fatalf("%s row %d score/bound (%v,%v), want (%v,%v)", metric, row, gotScore, gotBound, wantScore, wantBound)
			}
		}
		back, err := borrowed.MarshalBinary()
		if err != nil || !reflect.DeepEqual(back, data) {
			t.Fatalf("%s borrowed marshal: %v", metric, err)
		}
	}
}

// Every scalar a 1-bit score depends on prunes the exact rerank before a
// full vector is read, so bits the encoder could never write must not decode.
func TestBitCodesRejectImpossibleMetadata(t *testing.T) {
	vectors := corpus(4, 64)
	b, err := mustEncode(t, vectors, bitOpts("l2")).MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	norm0 := bitsHeader + 4*64
	align0 := norm0 + 4*4
	words0 := align0 + 4*4
	cases := map[string]func([]byte){
		"short":            func(d []byte) { copy(d, d[:len(d)-1]) },
		"bad magic":        func(d []byte) { copy(d, "XXXX") },
		"bad version":      func(d []byte) { binary.LittleEndian.PutUint16(d[4:], 9) },
		"bad metric byte":  func(d []byte) { d[6] = 3 },
		"other query bits": func(d []byte) { d[7] = 8 },
		"zero seed":        func(d []byte) { binary.LittleEndian.PutUint64(d[16:], 0) },
		"NaN centroid":     func(d []byte) { binary.LittleEndian.PutUint32(d[bitsHeader:], math.Float32bits(float32(math.NaN()))) },
		"negative norm":    func(d []byte) { binary.LittleEndian.PutUint32(d[norm0:], math.Float32bits(-1)) },
		"zero norm":        func(d []byte) { binary.LittleEndian.PutUint32(d[norm0:], 0) },
		"alignment above one": func(d []byte) {
			binary.LittleEndian.PutUint32(d[align0:], math.Float32bits(1.5))
		},
		"alignment below the sign floor": func(d []byte) {
			binary.LittleEndian.PutUint32(d[align0:], math.Float32bits(0.01))
		},
		"directionless in an l2 hood": func(d []byte) {
			binary.LittleEndian.PutUint32(d[align0:], math.Float32bits(zeroRowAlign))
		},
		"bits on a row with no residual": func(d []byte) {
			binary.LittleEndian.PutUint32(d[norm0:], 0)
			binary.LittleEndian.PutUint32(d[align0:], 0)
			binary.LittleEndian.PutUint64(d[words0:], 1)
		},
	}
	for name, corrupt := range cases {
		bad := append([]byte(nil), b...)
		if name == "short" {
			bad = bad[:len(bad)-1]
		} else {
			corrupt(bad)
		}
		if got, err := UnmarshalBinary(bad); err == nil {
			t.Fatalf("%s: decoded %d rows", name, got.Rows())
		}
		if got, err := UnmarshalBinaryBorrowed(bad); err == nil {
			t.Fatalf("%s: borrowed decode accepted %d rows", name, got.Rows())
		}
	}
	// 64 dims is a full word, so there is no padding to corrupt; check the
	// padding rule where there IS padding.
	pb, err := mustEncode(t, corpus(4, 33), bitOpts("l2")).MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	bad := append([]byte(nil), pb...)
	off := bitsHeader + 4*33 + 8*4
	binary.LittleEndian.PutUint64(bad[off:], binary.LittleEndian.Uint64(bad[off:])|1<<40)
	if _, err := UnmarshalBinary(bad); err == nil {
		t.Fatal("accepted a bit past dimension 33")
	}
}

// The estimator is the paper's: unbiased in the inner product, and so
// unbiased in the squared distance it reconstructs. A query's rows share its
// rotation and its 4-bit rounding, so their errors are NOT independent and
// the unit of evidence is the QUERY: take each query's mean error, and ask
// whether the mean of those is inside the standard error of their own
// spread. Anything else would call sampling noise a bias.
func TestBitEstimatorIsUnbiased(t *testing.T) {
	for _, metric := range []string{"l2", "cosine"} {
		vectors := corpus(2000, 128)
		queries := corpus(60, 128)
		codes := mustEncode(t, vectors, bitOpts(metric))
		perQuery := make([]float64, len(queries))
		var absSum, spread float64
		for i, q := range queries {
			s := codes.Scorer(NewQuery(q, metric))
			for row := range vectors {
				exact := float64(Score(metric, q, vectors[row]))
				err := float64(s.Score(row)) - exact
				perQuery[i] += err / float64(len(vectors))
				absSum += math.Abs(err)
				spread += math.Abs(exact)
			}
		}
		n := float64(len(queries))
		var mean float64
		for _, x := range perQuery {
			mean += x / n
		}
		var variance float64
		for _, x := range perQuery {
			variance += (x - mean) * (x - mean) / (n - 1)
		}
		stderr := math.Sqrt(variance / n)
		scale := spread / (n * float64(len(vectors)))
		t.Logf("%s: mean error %.5g +/- %.5g (%.3f%% of the mean |score|), mean |error| %.2f%%",
			metric, mean, stderr, 100*mean/scale, 100*absSum/(n*float64(len(vectors)))/scale)
		if math.Abs(mean) > 3*stderr {
			t.Fatalf("%s: mean error %v is %.1f standard errors from zero", metric, mean, math.Abs(mean)/stderr)
		}
	}
}

// The bound is probabilistic, so the test is coverage: the share of rows
// whose exact score exceeds the bound must sit at the tail boundSigmas
// names, and not above it. An axis-aligned corpus is in the set because
// that is what the random rotation exists for — without it, a corpus whose
// energy sits on a few axes has a quantization error the bound does not
// describe.
func TestBitBoundCoversAtItsConfidence(t *testing.T) {
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
	} {
		for _, metric := range []string{"l2", "cosine"} {
			codes := mustEncode(t, tc.v, bitOpts(metric))
			// Two settings of eps0 two orders of magnitude of tail apart:
			// the shipped one, and a loose one where a miscalibrated bound
			// would show up as a coverage that does not follow it.
			for _, sigmas := range []float64{2, boundSigmas} {
				var over, total int
				for _, q := range tc.q {
					query := NewQuery(q, metric)
					query.Sigmas = sigmas
					s := codes.Scorer(query)
					for row := range tc.v {
						_, bound := s.ScoreAndBound(row)
						if Score(metric, q, tc.v[row]) > bound {
							over++
						}
						total++
					}
				}
				share := float64(over) / float64(total)
				tail := 0.5 * math.Erfc(sigmas/math.Sqrt2) // one-sided normal
				t.Logf("%s %s at %.1f sigmas: %d/%d rows above the bound (%.5f against the %.5f tail)",
					tc.name, metric, sigmas, over, total, share, tail)
				// Conservative is required; wildly conservative would mean
				// the bound is not the confidence it claims, and would cost
				// the bound pass rows for nothing.
				if share > 4*tail {
					t.Fatalf("%s %s at %.1f sigmas: exceeded by %.5f of rows, past the %.5f it claims",
						tc.name, metric, sigmas, share, tail)
				}
				if sigmas == 2 && share < tail/20 {
					t.Fatalf("%s %s at %.1f sigmas: only %.5f of rows exceed a bound whose tail is %.5f; it is far wider than it claims",
						tc.name, metric, sigmas, share, tail)
				}
			}
		}
	}
}

// Query.Exact takes the probabilistic bound out of the picture: every row
// survives, so a bound-pruned pass over them is exact again.
func TestBitExactQueryBoundsNothing(t *testing.T) {
	vectors := corpus(200, 64)
	codes := mustEncode(t, vectors, Options{Metric: "l2", Seed: 3})
	s := codes.Scorer(Query{Vector: vectors[0], Metric: "l2", Exact: true})
	for row := range vectors {
		if _, bound := s.ScoreAndBound(row); !math.IsInf(float64(bound), 1) {
			t.Fatalf("row %d bound %v under Exact", row, bound)
		}
	}
}

// The two passes a query runs, in miniature:
// a shortlist of k x multiplier rows scored exactly, then every remaining row
// whose bound still reaches the cutoff. It is the bound pass, not the
// shortlist, that carries 1-bit recall — the shortlist alone is 0.5 at 2x
// here — and what it costs is rows.
//
// The fixture is deliberately the worst case an ANN index can be handed:
// isotropic Gaussian vectors have no structure at all, so the top ten sit in
// a crowd of near-ties and the error band covers a large share of them. Real
// corpora are not like this; SIFT1M behaves far better.
func TestBitTwoPassRecallAndRows(t *testing.T) {
	for _, metric := range []string{"l2", "cosine"} {
		vectors, queries := corpus(3000, 128), corpus(20, 128)
		for i := range queries {
			for d := range queries[i] {
				queries[i][d] += float32(i+d) * 0.0001 // held out
			}
		}
		{
			codes := mustEncode(t, vectors, bitOpts(metric))
			for _, mult := range []int{2, 5, 10} {
				var seedRecall, twoPass, rows float64
				for _, q := range queries {
					exact := func(i int) float32 { return Score(metric, q, vectors[i]) }
					want := top(vectors, q, exact, 10)
					s := codes.Scorer(NewQuery(q, metric))
					shortlist := top(vectors, q, s.Score, 10*mult)
					seedRecall += recall(Rerank(shortlist, exact, 10), want)
					seen := make(map[int]bool, len(shortlist))
					for _, row := range shortlist {
						seen[row] = true
					}
					read := append([]int(nil), shortlist...)
					cutoff := exact(Rerank(shortlist, exact, 10)[9])
					for row := range vectors {
						if _, bound := s.ScoreAndBound(row); !seen[row] && bound >= cutoff {
							read = append(read, row)
						}
					}
					rows += float64(len(read))
					twoPass += recall(Rerank(read, exact, 10), want)
				}
				n := float64(len(queries))
				t.Logf("%s %2dx: shortlist alone %.4f, with the bound pass %.4f, over %.0f rows (%.2f%%)",
					metric, mult, seedRecall/n, twoPass/n, rows/n, 100*rows/n/float64(len(vectors)))
				if r := twoPass / n; r < 0.99 {
					t.Fatalf("%s %dx: two-pass recall@10 %.4f", metric, mult, r)
				}
			}
		}
	}
}

// A zero vector has no direction: its cosine is 0, exactly, and the codec
// says so rather than estimating a distance on the unit sphere it is not on.
func TestBitZeroRowsAndZeroQueries(t *testing.T) {
	vectors := corpus(8, 16)
	vectors[3] = make([]float32, 16)
	codes := mustEncode(t, vectors, bitOpts("cosine"))
	s := codes.Scorer(NewQuery(vectors[0], "cosine"))
	if score, bound := s.ScoreAndBound(3); score != 0 || bound != 0 {
		t.Fatalf("zero row scored %v bound %v", score, bound)
	}
	zero := codes.Scorer(NewQuery(make([]float32, 16), "cosine"))
	for row := range vectors {
		if score, _ := zero.ScoreAndBound(row); score != 0 {
			t.Fatalf("zero query scored row %d at %v", row, score)
		}
	}
}

// A query that cannot be scored against these bits must PRUNE nothing: the
// bound pass then reads every row and the answer is still right. A bound of
// zero would silently drop the whole index instead.
func TestBitScorerBoundsNothingWhenItCannotScore(t *testing.T) {
	vectors := corpus(16, 32)
	l2 := mustEncode(t, vectors, bitOpts("l2"))
	cosine := mustEncode(t, vectors, bitOpts("cosine"))
	for name, s := range map[string]Scorer{
		"unknown metric":     l2.Scorer(NewQuery(vectors[0], "hamming")),
		"short query":        l2.Scorer(NewQuery(vectors[0][:8], "l2")),
		"cosine query on l2": l2.Scorer(NewQuery(vectors[0], "cosine")),
		"l2 query on cosine": cosine.Scorer(NewQuery(vectors[0], "l2")),
	} {
		score, bound := s.ScoreAndBound(3)
		if score != 0 || !math.IsInf(float64(bound), 1) {
			t.Fatalf("%s: scored %v bound %v, want 0 and +Inf", name, score, bound)
		}
	}
	// A zero cosine query is different: every score really IS 0, so 0 is a
	// sound bound and the rerank can skip on it.
	zero := cosine.Scorer(NewQuery(make([]float32, 32), "cosine"))
	if score, bound := zero.ScoreAndBound(3); score != 0 || bound != 0 {
		t.Fatalf("zero cosine query: scored %v bound %v, want 0 and 0", score, bound)
	}
}

// The rotation is reproducible from the seed alone and is an isometry —
// that is what lets the pack store eight bytes instead of a D x D matrix,
// and what makes ||o - c|| in the original frame the same number the bits
// were taken in.
func TestRotationIsReproducibleAndOrthogonal(t *testing.T) {
	const dims = 96
	v := corpus(1, dims)[0]
	a, b := make([]float32, dims), make([]float32, dims)
	copy(a, v)
	copy(b, v)
	newRotation(1234, dims).Apply(a)
	rotationFor(1234, dims).Apply(b)
	if !reflect.DeepEqual(a, b) {
		t.Fatal("the cached rotation is not the seeded one")
	}
	norm := func(x []float32) float64 {
		var n float64
		for _, f := range x {
			n += float64(f) * float64(f)
		}
		return math.Sqrt(n)
	}
	if got, want := norm(a), norm(v); math.Abs(got-want) > 1e-3*want {
		t.Fatalf("rotation changed the norm: %v != %v", got, want)
	}
	c := make([]float32, dims)
	copy(c, v)
	rotationFor(1235, dims).Apply(c)
	if reflect.DeepEqual(a, c) {
		t.Fatal("two seeds gave one rotation")
	}
	// The mixing the bound assumes: a coordinate vector must not survive as
	// a coordinate vector.
	e := make([]float32, dims)
	e[0] = 1
	rotationFor(7, dims).Apply(e)
	spread := 0
	for _, x := range e {
		if math.Abs(float64(x)) > 0.01 {
			spread++
		}
	}
	if spread < dims/2 {
		t.Fatalf("a basis vector rotated onto %d of %d coordinates", spread, dims)
	}
}

func BenchmarkBitScore(b *testing.B) {
	v := corpus(10000, 128)
	q := v[0]
	c, err := Quantize(v, Options{Metric: "l2", Seed: 11})
	if err != nil {
		b.Fatal(err)
	}
	b.Run("1bit", func(b *testing.B) {
		pq := NewQuery(q, "l2")
		for b.Loop() {
			s := c.Scorer(pq)
			for i := range v {
				_, _ = s.ScoreAndBound(i)
			}
		}
	})
	b.Run("f32", func(b *testing.B) {
		for b.Loop() {
			for i := range v {
				_ = L2Sq(q, v[i])
			}
		}
	})
}

// nextUp is math.Nextafter32 toward +Inf, bit for bit.
func TestNextUpIsNextafter32(t *testing.T) {
	inf := float32(math.Inf(1))
	xs := []float32{0, float32(math.Copysign(0, -1)), 1, -1, inf, -inf, float32(math.NaN()),
		math.MaxFloat32, -math.MaxFloat32, math.SmallestNonzeroFloat32, -math.SmallestNonzeroFloat32,
		math.Float32frombits(0x007fffff), math.Float32frombits(0x807fffff)}
	r := rand.New(rand.NewSource(1))
	for range 100000 {
		xs = append(xs, math.Float32frombits(r.Uint32()))
	}
	for _, x := range xs {
		got, want := nextUp(x), math.Nextafter32(x, inf)
		if math.Float32bits(got) != math.Float32bits(want) && !(got != got && want != want) {
			t.Fatalf("nextUp(%v) = %v, Nextafter32 %v", x, got, want)
		}
	}
}
