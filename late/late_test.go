package late

import (
	"github.com/axiomhq/ivfq"
	"math"
	"testing"
)

func TestLateInteractionPreservesTokenMatches(t *testing.T) {
	query := [][]float32{{1, 0}, {0, 1}}
	separate := [][]float32{{1, 0}, {0, 1}, {-1, -1}, {-1, -1}}
	pooled := [][]float32{{1, 1}}
	if got := Score(ivfq.Cosine, query, separate); got != 2 {
		t.Fatalf("individual perfect matches scored %v, want 2", got)
	}
	if Score(ivfq.Cosine, query, separate) <= Score(ivfq.Cosine, query, pooled) {
		t.Fatal("late interaction must prefer the separate perfect token matches")
	}
	// Averaging the same embeddings reverses that ranking.
	if ivfq.Score(ivfq.Cosine, []float32{0.5, 0.5}, []float32{-0.25, -0.25}) >= ivfq.Score(ivfq.Cosine, []float32{0.5, 0.5}, []float32{1, 1}) {
		t.Fatal("fixture does not distinguish token scoring from average pooling")
	}
	if got := Score(ivfq.Cosine, [][]float32{{1, 0}, {1, 0}}, [][]float32{{-1, 0}, {-1, 0}}); got != -2 {
		t.Fatalf("negative maxima were lost or averaged: %v", got)
	}
	if got := Score(ivfq.L2, [][]float32{{0}, {10}, {20}}, [][]float32{{1}, {9}}); got != -123 {
		t.Fatalf("variable-token sum of closest squared distances: %v, want -123", got)
	}
}

func TestLateInteractionRejectsInvalidMatrices(t *testing.T) {
	for name, matrix := range map[string][][]float32{
		"empty":       nil,
		"empty token": {{}},
		"ragged":      {{1, 2}, {3}},
		"wrong dims":  {{1}},
		"nan":         {{float32(math.NaN()), 1}},
		"infinity":    {{0, float32(math.Inf(-1))}},
	} {
		t.Run(name, func(t *testing.T) {
			if err := Validate(matrix, 2); err == nil {
				t.Fatal("invalid token matrix accepted")
			}
		})
	}
}
