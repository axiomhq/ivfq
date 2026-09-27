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
| types | `Vector[T]`, `Rows[T]`, `Sparse[T]`, `Metric` (`L2`, `Cosine`, `InnerProduct`) |

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
