//go:build race

package ivfq

// raceEnabled: the race detector makes sync.Pool drop items at random, so
// allocation counts of pooled paths mean nothing under it.
const raceEnabled = true
