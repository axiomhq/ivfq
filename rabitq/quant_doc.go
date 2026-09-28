// The quantization half provides a hood's candidate column and the exact rerank it
// seeds. There is ONE codec — 1-bit RaBitQ (rabitq.go): the row's residual
// to its hood's own mean, rotated into a random orthogonal frame and kept as
// one bit per dimension plus two scalars. A row costs dims BITS plus 8
// bytes, and its bound is a stated confidence rather than a certainty; a
// caller that wants certainty back names Query.Exact, which drops the bound
// and reranks every row of every probed hood exactly.
//
// The per-vector symmetric int8 codec (DQC2) was the other one until
// 2026-09-15 and is gone. The
// codec is an implementation choice, distinct from ANN v3's documented
// binary RaBitQ only in what the bound is used for, and distinct from
// user-visible stored vector precision, which the exact rerank reads.

package rabitq
