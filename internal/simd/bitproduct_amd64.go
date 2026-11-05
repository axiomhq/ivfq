package simd

// bitProductAVX2 is bitProductGeneric over words words at row and the four
// planes at planes, planes+words, ... : 32 bytes at a time with the
// nibble-table popcount (VPSHUFB + VPSADBW), the last words%4 with POPCNTQ.
//
//go:noescape
func bitProductAVX2(row *byte, planes *uint64, words int) (ones, weighted uint64)
