package ivfq

import "testing"

func TestWrapRowDoesNotCopy(t *testing.T) {
	words := []uint64{0}
	s := wrapRow(words)
	s.Set(3)
	if words[0] != 1<<3 {
		t.Fatalf("wrap copied or missed the write: %x", words[0])
	}
	other := wrapRow([]uint64{1 << 3})
	if s.IntersectionCardinality(other) != 1 {
		t.Fatalf("IntersectionCardinality = %d", s.IntersectionCardinality(other))
	}
}
