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
