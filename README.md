# ivfq

```sh
go get github.com/axiomhq/ivfq
```

Inverted-file clustering plus 1-bit quantization for approximate nearest
neighbour search, in pure Go with AVX2 kernels. The FAISS "IVF + quantizer"
pair, without the storage: you keep the rows, this package gives you the
clusters, the codes and the scan.

| package | what it holds |
| --- | --- |
| `ivfq` | sizing rule (`HoodK`, `DefaultNprobe`), `Policy`, split and merge rules, the centroid `Tree`, the centroid codec, and the distance vocabulary (`Metric`, `Score`, `Dot`, `L2Sq`, `CosineSim`) |
| `kmeans` | `Config.Fit`: sampled greedy k-means++ with one parallel assignment pass |
| `rabitq` | 1-bit RaBitQ codes: `Quantize`, `Quantizer`, `Scorer`, `Rotation` |
| `rank` | top-k selection and reciprocal rank fusion |
| `late` | late-interaction scoring of multi-vector documents |
| `sparse` | the string-keyed sparse vector |
| `codec` | f16 and int8 row storage |
| `bench` | ANN corpus readers and measurement helpers |

No I/O anywhere but `bench`.

## Build an index

```go
k := ivfq.HoodK(len(vectors)) // clusters of about 1,024 rows
centroids, assign, err := kmeans.Config{K: k, Iters: 11, Seed: 1}.Fit(ctx, vectors)
tree, err := ivfq.Build(ctx, centroids, 32)
rot := rabitq.NewRotation(rabitq.Seed("index", "field"), dims)
codes, err := rabitq.Quantize(rowsOfCluster, rabitq.Options{Metric: ivfq.L2, Rotation: rot})
```

`Fit` trains on a sample of at most `kmeans.SampleSize(k)` rows and assigns
every row in one parallel pass; the result is bit-identical whatever
`GOMAXPROCS` is. To route rows later, `s := tree.NewSearcher()` then
`s.Assign(row, beam)`, one `Searcher` per goroutine. One bit per dimension
plus two float32 per row.

## Query

```go
probe := tree.Nearest(query, ivfq.DefaultNprobe(k))
sc := codes.Scorer(rabitq.NewQuery(query, ivfq.L2, rot))
score, bound := sc.ScoreAndBound(row)   // estimate and upper bound
exact := ivfq.Score(ivfq.L2, query, raw) // rerank the shortlist
```

A row may be skipped by an exact top-k cutoff only when its bound is below
the cutoff. The bound is probabilistic; `Query.Exact` drops it.

Splits and merges keep the clusters at their size as rows arrive: a split
is `kmeans.Config{K: 2, Iters: kmeans.SplitIters}`, and `Policy`,
`MergeTarget`, `SplitAbove` and `MergeBelow` decide when, reading only
per-cluster counts and radii. No rule ever touches every row.

## Rank

`rank` orders hits by score descending, ties by id ascending.

1. Select the top k as you score: `sel := rank.NewTopK(k)`, `sel.Push(id, score)` per candidate, then `sel.Hits()`. `rank.Select(hits, k)` does the same for a slice.
2. Fuse ranked legs: `rank.RRF(k, legs...)`, or `rank.Fusion{RankConstant: c, Weights: w}.RRF(k, legs...)` after `Validate(len(legs))`.
3. Late interaction: `late.Score(metric, queryTokens, docTokens)` sums, over query tokens, the best document-token score. Check inputs with `late.Validate` first.

## Benchmark data

`bench` reads the standard ANN corpora and measures a run. Stdlib only.

1. Read a corpus: `bench.ReadFvecs(path)` for texmex `.fvecs`, or `b, err := bench.OpenBin(path)` then `b.Next(n)` in batches for Big ANN `.u8bin`/`.fbin`.
2. Read the ground truth: `bench.ReadIvecs(path)` for `.ivecs`, `bench.ReadGroundTruth(path)` for `.ibin`.
3. Score: `bench.RecallAtK(got, truth, 10)`.
4. Measure: `bench.Summary("query", latencies)` prints p50/p95/p99/mean; `m := bench.StartMemPeak()` ... `m.Close()` keeps peak `Sys`, `Heap` and `RSS`.

## Kernels

Every SIMD kernel lives in `internal/simd` with a pure Go reference and a
test that pins it. amd64 picks the kernel at run time from
`golang.org/x/sys/cpu`; every other architecture, and amd64 without the
flags, runs the Go reference.

| kernel | amd64 path | needs | claim | test |
| --- | --- | --- | --- | --- |
| `Dots` | `dotsAVX2FMA` | AVX2 + FMA | each `out[r]` equals `Dot(q, row r)` bit for bit | `TestDotsMatchesDot` |
| 1-bit scan | `bitProductAVX2` | AVX2 + POPCNT | popcounts identical to the Go loop | `TestBitProductMatchesGeneric` |
| rotation | `rotatePairsAVX2` | AVX2 | no FMA, so equal to the Go loop bit for bit | `TestRotatePairsMatchesGeneric` |
| i8 codec | `decodeI8AVX2`, `encodeI8AVX2` | AVX2 | same bytes and floats as the scalar codec | `TestI8CodecsMatchScalar` |

`Dot`, `L2Sq` and `CosineSim` keep several float32 accumulator lanes, so they
can differ from a float64 loop by about 1e-7 relative. Rankings do not change,
but centroids are only reproducible against a fixed kernel: treat a kernel
change like a seed change.

## Benchmark

`BenchmarkBitScore`, 10,000 rows of 128 dimensions, AMD EPYC 7502P (AVX2),
Go 1.27, n=10:

| | per 10k rows | per row |
| --- | --: | --: |
| 1-bit scan, estimate and bound | 427.6 µs ± 1% | 43 ns |
| f32 `L2Sq` over raw rows | 258.9 µs ± 2% | 26 ns |

## Test

```sh
go test -race ./...
go test -run '^$' -bench BitScore -count 10 ./rabitq
```

## License

MIT, see [LICENSE](LICENSE).
