// Package rank orders scored hits: score descending, ties by id ascending.
//
// TopK selects the k best of a stream of candidates with a bounded min-heap,
// Select does the same for a slice, and RRF fuses several rankings by
// reciprocal rank fusion. No I/O.
package rank
