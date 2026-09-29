package bench

import (
	"testing"
)

func TestRecallAtK(t *testing.T) {
	truth := [][]int32{{1, 2, 3, 4}, {5, 6, 7, 8}}
	got := [][]int32{{3, 1, 9, 4}, {5}}
	// Query 0: {3, 1} of truth[:2] = {1, 2} → 1 of 2. Query 1: {5} → 1 of 2.
	if r := RecallAtK(got, truth, 2); r != 0.5 {
		t.Fatalf("recall@2 = %v, want 0.5", r)
	}
	// k past both rows: query 0 finds 1, 3, 4 of 4; query 1 finds 5.
	if r := RecallAtK(got, truth, 4); r != (3.0/4+1.0/4)/2 {
		t.Fatalf("recall@4 = %v", r)
	}
	if r := RecallAtK(nil, truth, 10); r != 0 {
		t.Fatalf("no queries: %v", r)
	}
}
