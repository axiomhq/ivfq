# ivfq

```sh
go get github.com/axiomhq/ivfq
```

Inverted-file clustering plus 1-bit quantization for approximate nearest
neighbour search, in pure Go with AVX2 kernels. The FAISS "IVF + quantizer"
pair, without the storage: you keep the rows, this package gives you the
clusters, the codes and the scan.

## Build an index

1. Train centroids on a sample: `centroids, _, err := ivfq.RunSampledBudget(ctx, vectors, k, iters, seed, bytes)`.
   `k := ivfq.HoodK(n)` sizes clusters at about 1,024 rows.
2. Put them in a tree: `tree, err := ivfq.Build(ctx, centroids, fanout)`.
3. Route every row to its cluster: `tree.Nearest(row, beam)[0]`.
4. Quantize each cluster's rows: `q, err := ivfq.Quantize(rows, ivfq.Options{Dims: dims, Seed: seed})`.
   One bit per dimension plus two float32 per row.

## Query

1. Probe the nearest clusters: `tree.Nearest(query, ivfq.DefaultNprobe(k))`.
2. Score every probed row on its codes: `s := q.Scorer(ivfq.NewQuery(query, "l2"))`; `s.ScoreAndBound(row)` returns an estimate and a bound.
3. Rerank the shortlist on the raw vectors with `ivfq.Score(metric, query, row)`.

Splits and merges keep the clusters at their size as rows arrive:
`Split`, `SplitAbove`, `MergeBelow`, `Tree.Upsert`. No verb ever touches every
row; the policy (`Policy`, `DefaultPolicy`) only reads per-cluster counts and
radii.

## What is in it

| area | names |
| --- | --- |
| clustering | `RunSampled`, `RunSampledSpherical`, `RunSampledBudget`, `SampleSize`, `Split`, `SplitSpherical`, `SplitAbove`, `MergeBelow` |
| centroid tree | `Build`, `Tree`, `Tree.Nearest`, `Tree.Upsert`, `MarshalBinary`, `UnmarshalTree` |
| sizing | `HoodK`, `DefaultNprobe`, `Policy`, `DefaultPolicy` |
| quantizer | `Quantize`, `Quantizer`, `Code`, `Query`, `Scorer`, `Scorer.ScoreAndBound`, `Rotation` |
| codecs | `AppendF16`, `DecodeF16Into` (round to nearest even), `AppendI8`, `DecodeI8Into` (half away from zero, saturating) |
| kernels | `Dot`, `L2Sq`, `CosineSim`, `Score`, `Dots` |
| scoring | `LateInteractionScore`, `ValidateLateInteraction` |
| types | `Vector[T]`, `Rows[T]`, `Sparse[T]`, `Metric` (`L2`, `Cosine`, `InnerProduct`) |
| recall tuning (`recall`) | `New`, `Controller`, `Offer`, `Due`, `Measure`, `Tuned`, `Set`, `Source`, `Knobs`, `Depth`, `Overlap` |
| rank (`rank`) | `Hit`, `TopK`, `Select`, `Fusion`, `RRF` |

## Rank

`github.com/axiomhq/ivfq/rank` orders hits by score descending, ties by id ascending.

1. Select the top k as you score: `sel := rank.NewTopK(k)`, `sel.Push(id, score)` per candidate, then `sel.Hits()`. `rank.Select(hits, k)` does the same for a slice.
2. Fuse ranked legs: `rank.RRF(k, legs...)`, or `rank.Fusion{RankConstant: c, Weights: w}.RRF(k, legs...)` after `Validate(len(legs))`.
3. Late interaction: `ivfq.LateInteractionScore(metric, queryTokens, docTokens)` sums, over query tokens, the best document-token score. Check inputs with `ValidateLateInteraction` first.

## Tune recall

`github.com/axiomhq/ivfq/recall` holds an index at a recall@k target by
moving two knobs from a sample of live queries.

1. Build one controller: `c := recall.New(recall.Config{})` (target 0.95, 1 in 100 queries, 30 s between measurements).
2. Run each query at `c.Tuned(recall.Key{Index: name, Field: field})`.
3. After it, if `c.Offer(name)` and `c.Due(name)`, call `c.Measure(ctx, key, src)`, in the background if you like.

What it tunes:

| knob | range | order |
| --- | --- | --- |
| `Knobs.Probes` | `MinProbes` (16) to 2 × `DefaultNprobe` | spent first, bought back second |
| `Knobs.Depth`, clusters the bound-pruned rerank reads | `None`, 8, 16, 32, 64, `Unbounded` | bought back first, spent second |

What `src` (a `recall.Source`, one sampled query on one view) gives it:

| method | returns |
| --- | --- |
| `Clusters()` | cluster count, and how many are cached |
| `Search(ctx, knobs)` | top-k ids at those knobs, and whether it answered exactly anyway |
| `Exact(ctx)` | the true top-k ids |
| `Reranked()` | rows live queries' rerank read since the last call |

Guarantees:

- Never below the floor: probes stay at or above 16, and a move that drops
  recall more than 0.02 under the target is undone and its value pinned as
  a floor for good.
- Only tunes on warm data: under 99% of clusters cached, `Measure` returns
  `ErrCold` before either arm runs.
- One knob per move, at least `MoveSamples` (5) samples apart, so each
  sample measures one change.
- Filtered queries tune probes on their own and keep an unbounded depth.
- No I/O, no goroutines. State lives in memory; `Set` restores it.

K-means seeds with greedy k-means++ and assigns in one parallel pass; the
result is bit-identical whatever `GOMAXPROCS` is.

## Kernels

Every SIMD kernel has a pure Go reference and a test that pins it. amd64
picks the kernel at run time from `golang.org/x/sys/cpu`; every other
architecture, and amd64 without the flags, runs the Go reference.

| kernel | amd64 path | needs | claim | test |
| --- | --- | --- | --- | --- |
| `Dots` | `dotsAVX2FMA` | AVX2 + FMA | each `out[r]` equals `Dot(q, row r)` bit for bit | `TestDotsMatchesDot` |
| 1-bit scan | `bitProductAVX2` | AVX2 + POPCNT | popcounts identical to the Go loop | `TestBitProductMatchesGeneric` |
| rotation | `rotatePairsAVX2` | AVX2 | no FMA, so equal to the Go loop bit for bit | `TestRotatePairsMatchesApply` |
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
go test -run '^$' -bench BitScore -count 10 .
```

## License

MIT, see [LICENSE](LICENSE).
