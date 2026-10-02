#include "textflag.h"

// func dotsI8AVX2(q *int8, block *int8, dims, wide, rows int, out *int32)
TEXT ·dotsI8AVX2(SB), NOSPLIT, $0-48
	MOVQ q+0(FP), AX
	MOVQ block+8(FP), BX
	MOVQ dims+16(FP), CX
	MOVQ wide+24(FP), R10
	MOVQ rows+32(FP), DX
	MOVQ out+40(FP), R9

row:
	VPXOR Y0, Y0, Y0
	XORQ DI, DI

lanes:
	VPMOVSXBW (AX)(DI*1), Y1
	VPMOVSXBW (BX)(DI*1), Y2
	VPMADDWD Y2, Y1, Y3
	VPADDD Y3, Y0, Y0
	ADDQ $16, DI
	CMPQ DI, R10
	JLT  lanes

	VEXTRACTI128 $1, Y0, X1
	VPADDD X1, X0, X0
	VPSHUFD $0x4e, X0, X1
	VPADDD X1, X0, X0
	VPSHUFD $0xb1, X0, X1
	VPADDD X1, X0, X0
	VMOVD X0, (R9)
	ADDQ CX, BX
	ADDQ $4, R9
	DECQ DX
	JNZ  row

	VZEROUPPER
	RET
