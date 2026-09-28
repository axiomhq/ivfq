#include "textflag.h"

// func dotsAVX2FMA(q *float32, block *float32, dims int, rows int, out *float32)
// out[r] = Σ q[i]*block[r*dims+i], each row with go-simd/floats'
// dot32AVX2 sequence: one YMM accumulator fused-multiply-added eight lanes
// at a time, folded high-onto-low and twice horizontally, then a scalar
// fused tail.
TEXT ·dotsAVX2FMA(SB), NOSPLIT, $0-40
	MOVQ q+0(FP), AX
	MOVQ block+8(FP), BX
	MOVQ dims+16(FP), CX
	MOVQ rows+24(FP), DX
	MOVQ out+32(FP), R9
	TESTQ DX, DX
	JZ   done

row:
	VXORPS Y0, Y0, Y0
	XORQ DI, DI

vloop:
	LEAQ 8(DI), R8
	CMPQ R8, CX
	JGT  vtail
	VMOVUPS (AX)(DI*4), Y1
	VMOVUPS (BX)(DI*4), Y2
	VFMADD231PS Y2, Y1, Y0
	ADDQ $8, DI
	JMP  vloop

vtail:
	VEXTRACTF128 $1, Y0, X1
	VADDPS X1, X0, X0
	VHADDPS X0, X0, X0
	VHADDPS X0, X0, X0

sloop:
	CMPQ DI, CX
	JGE  next
	VMOVSS (AX)(DI*4), X1
	VMOVSS (BX)(DI*4), X2
	VFMADD231SS X2, X1, X0
	ADDQ $1, DI
	JMP  sloop

next:
	VMOVSS X0, (R9)
	LEAQ (BX)(CX*4), BX
	ADDQ $4, R9
	DECQ DX
	JNZ  row

done:
	VZEROUPPER
	RET
