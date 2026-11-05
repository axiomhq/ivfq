package simd

import "golang.org/x/sys/cpu"

// What the kernels need, read once. The tests clear a flag to run the Go
// reference on an amd64 host.
var (
	hasAVX2       = cpu.X86.HasAVX2
	hasAVX2POPCNT = cpu.X86.HasAVX2 && cpu.X86.HasPOPCNT
	hasAVX2FMA    = cpu.X86.HasAVX2 && cpu.X86.HasFMA
	hasAVX512F    = cpu.X86.HasAVX512F
	// VNNI needs AVX-512 BW for the byte masks and F for the 512-bit registers.
	hasVNNI = cpu.X86.HasAVX512F && cpu.X86.HasAVX512BW && cpu.X86.HasAVX512VNNI
)
