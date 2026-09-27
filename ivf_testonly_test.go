package ivfq

import "context"

// Test-only helpers: nothing outside this package's tests calls them.

// Run clusters vecs into k groups (k clamped to [1, len(vecs)]) with a fixed
// iteration cap. Deterministic for a given seed — including across worker
// counts, see assignAll. Empty clusters are reseeded from the vector
// farthest from its assigned centroid. A cancelled ctx returns its error
// and no result: checked at every seeding step, every Lloyd iteration,
// and every assignment chunk.
func Run(ctx context.Context, vecs [][]float32, k, iters int, seed int64) ([][]float32, []int, error) {
	return run(ctx, vecs, k, iters, seed, false)
}

// RunSpherical is Run for unit-vector corpora (cosine namespaces):
// centroids are re-normalized to unit length after every mean
// recomputation, keeping centroid geometry on the sphere. Plain L2
// against unit centroids is then monotone in cosine, so assignment and
// query-time selection stay L2 everywhere (spherical k-means). A
// zero-norm mean (degenerate antipodal cluster) is left unnormalized.
func RunSpherical(ctx context.Context, vecs [][]float32, k, iters int, seed int64) ([][]float32, []int, error) {
	return run(ctx, vecs, k, iters, seed, true)
}
