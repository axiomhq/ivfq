package ivfq

import (
	"fmt"
	"math"
)

// ValidateLateInteraction checks a nonempty matrix of finite, fixed-width tokens.
func ValidateLateInteraction(tokens [][]float32, dims int) error {
	if dims <= 0 || len(tokens) == 0 {
		return fmt.Errorf("late-interaction requires at least one token and positive dimensions")
	}
	for i, token := range tokens {
		if len(token) != dims {
			return fmt.Errorf("late-interaction token %d has %d dims, expected %d", i, len(token), dims)
		}
		for _, value := range token {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return fmt.Errorf("late-interaction token %d contains a nonfinite value", i)
			}
		}
	}
	return nil
}

// LateInteractionScore sums each query token's best document-token score.
// Inputs must be validated nonempty matrices of the same width. In particular,
// a negative best match contributes its negative score, not zero.
func LateInteractionScore(metric string, query, document [][]float32) float32 {
	var sum float32
	for _, q := range query {
		best := float32(math.Inf(-1))
		for _, token := range document {
			if score := Score(metric, q, token); score > best {
				best = score
			}
		}
		sum += best
	}
	return sum
}
