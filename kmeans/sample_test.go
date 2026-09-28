package kmeans

import (
	"context"
	"math/rand"
	"testing"
)

// TestRunSampledBoundsTraining is the scaling rule as an assertion: Lloyd's
func TestRunSampledBoundsTraining(t *testing.T) {
	const n, dims, k = 50_000, 8, 64
	rng := rand.New(rand.NewSource(3))
	vecs := make([][]float32, n)
	for i := range vecs {
		v := make([]float32, dims)
		for d := range v {
			v[d] = float32(rng.NormFloat64()) + float32(i%16)
		}
		vecs[i] = v
	}
	centroids, assign, _ := (Config{K: k, Iters: 10, Seed: 1}).Fit(context.Background(), vecs)
	// RunSampled hands Lloyd's exactly sample(SampleSize(k)) and nothing
	// else: the centroids are the fit over that sample, bit for bit.
	trained, err := sample(context.Background(), vecs, SampleSize(k), 1)
	if err != nil || len(trained) != SampleSize(k) {
		t.Fatalf("sample = %d rows, %v; want SampleSize(%d) = %d", len(trained), err, k, SampleSize(k))
	}
	fit, _, _ := Run(context.Background(), trained, k, 10, 1)
	for i := range centroids {
		for d := range centroids[i] {
			if centroids[i][d] != fit[i][d] {
				t.Fatalf("RunSampled centroid %d dim %d = %v, fit over the sample = %v", i, d, centroids[i][d], fit[i][d])
			}
		}
	}
	if len(centroids) != k || len(assign) != n {
		t.Fatalf("centroids=%d assign=%d, want %d and %d", len(centroids), len(assign), k, n)
	}
	// Every vector assigned, and assigned to its true nearest — the sample
	// picks the frame, the full pass places the corpus.
	for i := range assign {
		if assign[i] < 0 || assign[i] >= k {
			t.Fatalf("assign[%d] = %d out of range", i, assign[i])
		}
		if i%997 == 0 {
			if want := Nearest(centroids, vecs[i]); assign[i] != want {
				t.Fatalf("assign[%d] = %d, want nearest %d", i, assign[i], want)
			}
		}
	}
	// A corpus the sample already covers gets the full fit, unchanged.
	small := vecs[:SampleSize(k)]
	got, _, _ := (Config{K: k, Iters: 10, Seed: 1}).Fit(context.Background(), small)
	want, _, _ := Run(context.Background(), small, k, 10, 1)
	for i := range got {
		for d := range got[i] {
			if got[i][d] != want[i][d] {
				t.Fatalf("RunSampled != Run when the sample covers the corpus (centroid %d, dim %d)", i, d)
			}
		}
	}
}

func TestSampleSizeForBudget(t *testing.T) {
	if got := SampleSizeForBudget(1000, 128, 1<<20); got > (1<<20)/(128*4) {
		t.Fatalf("sample has %d vectors, exceeds byte budget", got)
	}
	if got := SampleSizeForBudget(1000, 128, 1); got != 1 {
		t.Fatalf("minimum sample = %d, want 1", got)
	}
}
