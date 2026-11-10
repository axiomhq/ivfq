# ivfq

```sh
go get github.com/axiomhq/ivfq
```

Inverted-file clustering plus 1-bit quantization for approximate nearest
neighbour search, in Go with AVX2 assembly kernels and pure-Go fallbacks. The FAISS "IVF + quantizer"
pair, without the storage: you keep the rows, this package gives you the
clusters, the codes and the scan.

| package | what it holds |
| --- | --- |
| `ivfq` | sizing rule (`HoodK`, `DefaultNprobe`), `Policy`, split and merge rules, the centroid trees (`FloatTree`; `Tree[B]` over any `EmbeddingBatch`, such as `Dense[T]`), the centroid codec, and the distance vocabulary (`Metric`, `Embedding`, `EmbeddingBatch`, `Score`, `Dot`, `L2Sq`, `CosineSim`) |
| `kmeans` | `Config.Fit`: sampled greedy k-means++ with one parallel assignment pass, two-level above `kmeans.TwoLevelK` centroids; `Config.FitBalanced` for splits |
| `rabitq` | 1-bit RaBitQ codes, one bit per dimension plus two float32 per row: `Quantize`, `Quantizer`, `Scorer`, `Rotation`, and `Block` for `ivfq.Tree` |
| `rank` | top-k selection |
| `codec` | f16 and int8 row storage |
| `vector` | `Slice`: flat float32, f16, int8 or uint8 rows, read as `[][]float32` |
| `bench` | Big ANN corpus readers and recall@k |

No I/O anywhere but `bench`.

## Build an index

```go
k := ivfq.HoodK(len(vectors)) // clusters of about ivfq.HoodTarget (196) rows
centroids, assign, err := kmeans.Config{K: k, Iters: 11, Seed: 1}.Fit(ctx, vectors)
tree, err := ivfq.Build(ctx, centroids, 32)
rot := rabitq.NewRotation(rabitq.Seed("index", "field"), dims)
codes, err := rabitq.Quantize(rowsOfCluster, rabitq.Options{Metric: ivfq.L2, Rotation: rot})
```

`Fit` trains on a sample of at most `kmeans.SampleSize(k)` rows and assigns
every row in one parallel pass; the result is bit-identical whatever
`GOMAXPROCS` is. Above `kmeans.TwoLevelK` (512) it fits √k coarse
centroids, then k fine ones under them, and each row searches only its
16 nearest coarse centroids' fine ones: about 17·√k distances per row,
not k. To route rows later, `s := tree.NewSearcher()` then
`s.Assign(row, beam)`, one `Searcher` per goroutine.

`Build` returns a `FloatTree`: exact float32 centroids, `Leaf`, `Upsert`
and a wire format. For a smaller immutable router, `BuildTree` builds the
same hierarchy and stores every node's rows in a block you choose:

```go
small, err := ivfq.BuildTree(ctx, centroids, 32, ivfq.NewDense[int8])
bits, err := ivfq.BuildTree(ctx, centroids, 32, func(rows [][]float32) (rabitq.Block, error) {
	return rabitq.NewBlock(rows, rot)
})
q, err := ivfq.NewDense[float32]([][]float32{query}) // a dense query
probe, err := bits.Nearest(q.At(0), beam, ivfq.DefaultNprobe(k))
```

Every block is an `EmbeddingBatch`: `ScoreAll(query, metric, dst)` scores a
dense query (`DenseEmbedding` of any scalar) against all its rows, higher
is better, and `At(i)` is row i as an `Embedding`. `Dense` scores L2,
cosine and inner product of its decoded rows; `rabitq.Block` holds L2
codes and scores only the L2 estimate. A tree always routes by L2.

`Dense[float32]` routes exactly as `FloatTree`, except that a score
overflowing to infinity, which `FloatTree` ranks last, is an error.
`float16`, `int8` and `uint8` blocks round (integer blocks on one affine
scale per block), and `rabitq.Block` ranks by 1-bit estimates, so it misses some exact
neighbours and a dropped branch cannot be reranked back. A `Tree` keeps no
float32 copy of the centroids: keep your own if you need them.

## Query

```go
probe := tree.Nearest(query, ivfq.DefaultNprobe(k))
sc := codes.Scorer(rabitq.NewQuery(query, ivfq.L2, rot))
score, bound := sc.ScoreAndBound(row)   // estimate and upper bound
exact := ivfq.Score(ivfq.L2, query, raw) // rerank the shortlist
```

A row may be skipped by an exact top-k cutoff only when its bound is below
the cutoff. The bound is probabilistic; `Query.Exact` drops it.

For two-pass reranking, `Scorer.ScoreAllLazy(scores)` collects a `BoundHint`
while scoring. After exactly reranking the shortlist, call
`Scorer.RefineBounds(scores, bounds, cutoff, hint)` to avoid tight bounds for
obvious losers. Packed global L2 scoring uses this shortcut; other bindings
compute the original bounds. Keep the column unchanged and the cached scores
in the current query's row order. Refined bounds support a rising cutoff;
preserve the original rescue order and use strict `bound < cutoff` rejection.
`ScoreAll` still returns every original score and bound bit for bit. See the
executable `ExampleScorer_RefineBounds` in `rabitq/quant_test.go`.

Splits and merges keep the clusters at their size as rows arrive: a split
is `kmeans.Config{K: 2, Iters: kmeans.SplitIters}.FitBalanced`, which keeps
each child at least a quarter of the parent, and `Policy`,
`MergeTargets`, `SplitAbove` and `MergeBelow` decide when, reading only
per-cluster counts, radii and centroids. No rule ever touches every row.

## Rank

`rank` orders hits by score descending, NaN last, ties by id ascending.
Ids are ints: row or leaf numbers, as the rest of ivfq returns them.

Select the top k as you score: `sel := rank.NewTopK(k)`, `sel.Push(id, score)` per candidate, then `sel.Hits()`. `rank.Select(hits, k)` does the same for a slice.

## Benchmark data

`bench` reads the Big ANN corpora and scores recall. Stdlib only.

1. Read a corpus: `b, err := bench.OpenBin(path)`, `defer b.Close()`, then `b.Next(n)` in batches for `.u8bin`/`.i8bin`/`.fbin`. A sample cut from a corpus keeps the corpus header; read the rows you need, since reading past its end reports a truncation.
2. Read the ground truth: `bench.ReadGroundTruth(path)` for `.ibin`.
3. Score: `bench.RecallAtK(got, truth, 10)`.
4. CPU time: `bench.CPUSeconds()` before and after a run.

## Kernels

Every SIMD kernel lives in `internal/simd` beside a pure Go reference and a
test that pins it to that reference; amd64 picks the kernel at run time from
`golang.org/x/sys/cpu`, and every other architecture runs the reference.

`Dot`, `L2Sq` and `CosineSim` keep several float32 accumulator lanes, so they
can differ from a float64 loop by about 1e-7 relative. That can reorder near
ties, and a narrow beam that breaks a tie differently reaches different
leaves, so rankings and centroids are only reproducible against a fixed
kernel: treat a kernel change like a seed change.

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
