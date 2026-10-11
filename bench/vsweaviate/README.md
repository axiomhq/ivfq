# ivfq vs weaviate

Benchmarks of ivfq's kernels, 1-bit quantizer and 2-means against the weaviate
code they correspond to. A separate module, so weaviate's dependencies stay out
of ivfq's `go.mod`.

```sh
git clone --depth 1 https://github.com/weaviate/weaviate ../../../weaviate   # next to this repo
cd bench/vsweaviate
go test -run 'TestKernelsAgree'                  # kernel pairs compute the same value
go test -run 'TestEstimatorQuality|TestSplitQuality' -v    # recall / WCSS tables
go test -run '^$' -bench . -benchtime 200ms -count 5 -cpu 1
```

Measured against weaviate `0d36ea43` on a 2.6 GHz Xeon (AVX2, AVX-512, VNNI),
Go 1.27.1, one P. The VM is noisy: small ops move up to 2x between runs, so
read medians of `-count 5` and ignore differences under ~20%.

| file | compares |
|---|---|
| `kernels_test.go` | `internal/simd` + `go-simd/floats` vs `hnsw/distancer/asm`: f32 dot / L2, u8 dot / L2, f32×u8 dot, 1-bit plane scan |
| `rq_test.go` | `rabitq` vs `BinaryRotationalQuantizer`: rotation, encode, query bind, scan, recall |
| `kmeans_test.go` | `kmeans.Config{K:2}.Fit` vs `KMeans.FitBalanced` |

`weaviate/sroar` has no ivfq counterpart (ivfq has no bitmap type), so it is not benchmarked.
