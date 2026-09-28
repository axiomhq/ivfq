// Package ivfq is inverted-file clustering plus quantization: the two halves
// of an IVF index in the FAISS sense ("IVF + quantizer").
//
// The clustering half partitions vectors into hoods: k-means (plain,
// spherical, and sample-trained), the centroid Tree and its Nearest lookup,
// splits and merges, and the rebalancing Policy. The quantization half gives
// every hood a candidate column and an exact rerank: Vector[T], Rows,
// Sparse, Rotation, Quantizer, the 1-bit RaBitQ Code and its Scorer, and the
// SIMD kernels under them (Dot, L2Sq, CosineSim, Dots, the bit products).
// LateInteractionScore scores multi-vector documents token by token; package
// rank selects and fuses the results. Centroid sets and their slot deltas
// have a binary codec (EncodeCentroids, EncodeCentroidDelta), and
// SparseVector is the string-keyed sparse vector beside Sparse[T].
// No I/O.
package ivfq
