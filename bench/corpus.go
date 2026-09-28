package bench

import "math/rand"

// Blobs draws nBlobs gaussian centers (N(0, 10²) per dimension) and n
// vectors, vector i at centers[i%nBlobs] plus N(0, 1) noise. The same seed
// gives the same vectors.
func Blobs(seed int64, n, dims, nBlobs int) (vecs [][]float32, centers [][]float32) {
	rng := rand.New(rand.NewSource(seed))
	centers = blobCenters(rng, dims, nBlobs)
	vecs = make([][]float32, n)
	for i := range vecs {
		c := centers[i%nBlobs]
		v := make([]float32, dims)
		for d := range v {
			v[d] = c[d] + float32(rng.NormFloat64())
		}
		vecs[i] = v
	}
	return vecs, centers
}

// blobCenters draws the nBlobs cluster centers off rng.
func blobCenters(rng *rand.Rand, dims, nBlobs int) [][]float32 {
	centers := make([][]float32, nBlobs)
	for b := range centers {
		c := make([]float32, dims)
		for d := range c {
			c[d] = float32(rng.NormFloat64()) * 10
		}
		centers[b] = c
	}
	return centers
}

// RecallAtK is recall@k averaged over queries: for query i, the number of
// got[i][:k] ids found in truth[i][:k], over k. got[i] and truth[i] belong
// to the same query; truth has at least len(got) entries. A query that
// returned fewer than k ids, or a truth row shorter than k, scores below 1.
func RecallAtK(got, truth [][]int32, k int) float64 {
	if len(got) == 0 || k <= 0 {
		return 0
	}
	hits := 0.0
	want := make(map[int32]bool, k)
	for i, ids := range got {
		clear(want)
		for _, id := range truth[i][:min(k, len(truth[i]))] {
			want[id] = true
		}
		inter := 0
		for _, id := range ids[:min(k, len(ids))] {
			if want[id] {
				inter++
			}
		}
		hits += float64(inter) / float64(k)
	}
	return hits / float64(len(got))
}
