#include "textflag.h"

// popLUT<> is popcount(nibble) for the 16 nibbles, twice, for a 32-byte
// VPSHUFB; nib<> masks a byte to its low nibble.
DATA popLUT<>+0(SB)/8, $0x0302020102010100
DATA popLUT<>+8(SB)/8, $0x0403030203020201
DATA popLUT<>+16(SB)/8, $0x0302020102010100
DATA popLUT<>+24(SB)/8, $0x0403030203020201
GLOBL popLUT<>(SB), RODATA|NOPTR, $32
DATA nib<>+0(SB)/8, $0x0f0f0f0f0f0f0f0f
DATA nib<>+8(SB)/8, $0x0f0f0f0f0f0f0f0f
DATA nib<>+16(SB)/8, $0x0f0f0f0f0f0f0f0f
DATA nib<>+24(SB)/8, $0x0f0f0f0f0f0f0f0f
GLOBL nib<>(SB), RODATA|NOPTR, $32

// POPCNT32 adds the byte-wise popcount of Y0's contents masked into
// src (a Y register) to acc: per-byte nibble lookups summed into four
// qword lanes by VPSADBW. Y15 is the table, Y14 the nibble mask, Y13 zero;
// Y2 and Y3 are scratch.
#define POPCNT32(src, acc) \
	VPSRLW  $4, src, Y2       \
	VPAND   Y14, src, Y3      \
	VPAND   Y14, Y2, Y2       \
	VPSHUFB Y3, Y15, Y3       \
	VPSHUFB Y2, Y15, Y2       \
	VPADDB  Y3, Y2, Y2        \
	VPSADBW Y13, Y2, Y2       \
	VPADDQ  Y2, acc, acc

// HSUMQ folds acc's four qword lanes into the 64-bit register out.
#define HSUMQ(acc, accx, out) \
	VEXTRACTI128 $1, acc, X2  \
	VPADDQ  X2, accx, accx    \
	VPSRLDQ $8, accx, X2      \
	VPADDQ  X2, accx, accx    \
	VMOVQ   accx, out

// func bitProductAVX2(row *byte, planes *uint64, words int) (ones, weighted uint64)
TEXT ·bitProductAVX2(SB), NOSPLIT, $0-40
	MOVQ row+0(FP), AX
	MOVQ planes+8(FP), R8
	MOVQ words+16(FP), CX
	MOVQ CX, DX
	SHLQ $3, DX               // plane stride in bytes
	LEAQ (R8)(DX*1), R9
	LEAQ (R9)(DX*1), R10
	LEAQ (R10)(DX*1), R11

	VMOVDQU popLUT<>(SB), Y15
	VMOVDQU nib<>(SB), Y14
	VPXOR   Y13, Y13, Y13
	VPXOR   Y8, Y8, Y8        // ones
	VPXOR   Y9, Y9, Y9        // plane 1
	VPXOR   Y10, Y10, Y10     // plane 2
	VPXOR   Y11, Y11, Y11     // plane 3
	VPXOR   Y12, Y12, Y12     // plane 4

vloop:
	CMPQ CX, $4
	JL   vdone
	VMOVDQU (AX), Y0
	POPCNT32(Y0, Y8)
	VPAND (R8), Y0, Y1
	POPCNT32(Y1, Y9)
	VPAND (R9), Y0, Y1
	POPCNT32(Y1, Y10)
	VPAND (R10), Y0, Y1
	POPCNT32(Y1, Y11)
	VPAND (R11), Y0, Y1
	POPCNT32(Y1, Y12)
	ADDQ $32, AX
	ADDQ $32, R8
	ADDQ $32, R9
	ADDQ $32, R10
	ADDQ $32, R11
	SUBQ $4, CX
	JMP  vloop

vdone:
	HSUMQ(Y8, X8, SI)
	HSUMQ(Y9, X9, DI)
	HSUMQ(Y10, X10, R14)
	HSUMQ(Y11, X11, R15)
	HSUMQ(Y12, X12, R13)
	VZEROUPPER

tail:
	TESTQ CX, CX
	JZ    done
	MOVQ    (AX), R12
	POPCNTQ R12, DX
	ADDQ    DX, SI
	MOVQ    (R8), DX
	ANDQ    R12, DX
	POPCNTQ DX, DX
	ADDQ    DX, DI
	MOVQ    (R9), DX
	ANDQ    R12, DX
	POPCNTQ DX, DX
	ADDQ    DX, R14
	MOVQ    (R10), DX
	ANDQ    R12, DX
	POPCNTQ DX, DX
	ADDQ    DX, R15
	MOVQ    (R11), DX
	ANDQ    R12, DX
	POPCNTQ DX, DX
	ADDQ    DX, R13
	ADDQ $8, AX
	ADDQ $8, R8
	ADDQ $8, R9
	ADDQ $8, R10
	ADDQ $8, R11
	DECQ CX
	JMP  tail

done:
	MOVQ SI, ones+24(FP)
	SHLQ $1, R14
	SHLQ $2, R15
	SHLQ $3, R13
	ADDQ R14, DI
	ADDQ R15, DI
	ADDQ R13, DI
	MOVQ DI, weighted+32(FP)
	RET
