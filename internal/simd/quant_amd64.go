package simd

// blockSumSq is BlockSumSq: sums[0:16] += the squares of dims rows of 16.
//
//go:noescape
func blockSumSq(block *float32, dims int, sums *float64)

// blockBits is blockBits32 for n in 1..32: out[0:16] set to the sign bits,
// abs[0:16] += |value|.
//
//go:noescape
func blockBits(block *float32, n int, abs *float64, out *uint32)

// minMaxScaled is MinMaxScaled for n a multiple of 4, n >= 4.
//
//go:noescape
func minMaxScaled(x *float32, n int, mul float64) (lo, hi float64)

// sumSq is SumSq for n a multiple of 4.
//
//go:noescape
func sumSq(x *float32, n int) float64

// quantCodes is QuantCodes for n a multiple of 4.
//
//go:noescape
func quantCodes(codes *byte, x *float32, n int, mul, lo, delta, levels float64) float64

// scoreFast4 is ScoreFast for n a multiple of 4; cos 1 scores cosine.
//
//go:noescape
func scoreFast4(n int, sums, ones *uint16, norms, aligns, scores *float32, k *float64, cos int)

// subSumSq is SubSumSq's vector part for n a multiple of 8: the four lanes'
// sums, not yet combined.
//
//go:noescape
func subSumSq(x, c *float32, n int, lanes *[4]float64)

// planes32 is Planes for n a multiple of 32.
//
//go:noescape
func planes32(codes *byte, n int, planes *uint64, words int) uint64
