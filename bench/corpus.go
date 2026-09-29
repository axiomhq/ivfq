package bench

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
