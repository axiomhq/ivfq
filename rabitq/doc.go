// Package rabitq is the 1-bit candidate codec of an IVF index: RaBitQ
// (Gao & Long, SIGMOD 2024, arXiv:2405.12497). Each cluster's rows are
// encoded as the sign bits of their residual to the cluster's own
// centroid, rotated into a random orthogonal frame, plus two float32
// scalars per row: dims bits plus 8 bytes. A Scorer binds a query to a
// column and returns, per row, a higher-is-better estimate of the exact
// score and an upper bound on it that holds at a stated confidence;
// Query.Exact drops the bound for a caller that wants certainty and
// reranks every row exactly. Columns marshal to a versioned binary form.
package rabitq
