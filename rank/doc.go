// Package rank orders scored hits: score descending, NaN last, ties by id
// ascending.
//
// TopK selects the k best of a stream of candidates with a bounded min-heap,
// and Select does the same for a slice. No I/O.
package rank
