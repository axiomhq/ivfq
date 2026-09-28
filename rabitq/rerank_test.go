package rabitq

import "sort"

// Rerank scores candidate IDs exactly and returns the best k, with lower IDs
// winning equal scores. exactDistance must return a higher-is-better score.
func Rerank(candidates []int, exactDistance func(int) float32, k int) []int {
	r := make([]ranked, len(candidates))
	for i, id := range candidates {
		r[i] = ranked{id, exactDistance(id)}
	}
	sort.Slice(r, func(i, j int) bool { return r[i].score > r[j].score || r[i].score == r[j].score && r[i].id < r[j].id })
	k = min(max(k, 0), len(r))
	out := make([]int, k)
	for i := range out {
		out[i] = r[i].id
	}
	return out
}
