#include "textflag.h"

// Sum the sixteen float32 lanes of Z0 into X0.
#define REDUCE512 \
	VEXTRACTF64X4 $1, Z0, Y1 \
	VADDPS Y1, Y0, Y0 \
	VEXTRACTF128 $1, Y0, X1 \
	VADDPS X1, X0, X0 \
	VPSHUFD $0x4e, X0, X1 \
	VADDPS X1, X0, X0 \
	VPSHUFD $0xb1, X0, X1 \
	VADDPS X1, X0, X0

// func dotsF32AVX512(q *float32, block *float32, dims, rows int, out *float32)
// out[r] = q . row r, 64 lanes a step on four FMA chains, then 16 at a time,
// then one masked step. Dot calls it with rows = 1, so Dots is Dot bit for
// bit by construction.
TEXT ·dotsF32AVX512(SB), NOSPLIT, $0-40
	MOVQ q+0(FP), AX
	MOVQ block+8(FP), BX
	MOVQ dims+16(FP), CX
	MOVQ rows+24(FP), DX
	MOVQ out+32(FP), R9
	MOVQ CX, R11
	ANDQ $15, R11 // tail lanes
	JZ   nomask
	MOVQ $1, R12
	MOVQ R11, CX
	SHLQ CX, R12
	DECQ R12
	KMOVW R12, K1
	MOVQ dims+16(FP), CX
nomask:
	MOVQ CX, R8
	ANDQ $-16, R8 // lanes in whole vectors

row:
	VXORPS X0, X0, X0
	VXORPS X1, X1, X1
	VXORPS X2, X2, X2
	VXORPS X3, X3, X3
	XORQ DI, DI
	LEAQ 64(DI), SI
	CMPQ SI, R8
	JGT  one

quad:
	VMOVUPS (AX)(DI*4), Z4
	VMOVUPS 64(AX)(DI*4), Z5
	VMOVUPS 128(AX)(DI*4), Z6
	VMOVUPS 192(AX)(DI*4), Z7
	VFMADD231PS (BX)(DI*4), Z4, Z0
	VFMADD231PS 64(BX)(DI*4), Z5, Z1
	VFMADD231PS 128(BX)(DI*4), Z6, Z2
	VFMADD231PS 192(BX)(DI*4), Z7, Z3
	ADDQ $64, DI
	LEAQ 64(DI), SI
	CMPQ SI, R8
	JLE  quad

one:
	CMPQ DI, R8
	JGE  tail
	VMOVUPS (AX)(DI*4), Z4
	VFMADD231PS (BX)(DI*4), Z4, Z0
	ADDQ $16, DI
	JMP  one

tail:
	TESTQ R11, R11
	JZ    reduce
	VMOVUPS.Z (AX)(DI*4), K1, Z4
	VMOVUPS.Z (BX)(DI*4), K1, Z5
	VFMADD231PS Z5, Z4, Z0

reduce:
	VADDPS Z1, Z0, Z0
	VADDPS Z3, Z2, Z2
	VADDPS Z2, Z0, Z0
	REDUCE512
	VMOVSS X0, (R9)
	LEAQ (BX)(CX*4), BX
	ADDQ $4, R9
	DECQ DX
	JNZ  row

	VZEROUPPER
	RET

// func l2sqF32AVX512(a *float32, b *float32, n int) float32
// sum (a[i]-b[i])^2: no sqrt, same shape as the dot kernel.
TEXT ·l2sqF32AVX512(SB), NOSPLIT, $0-28
	MOVQ a+0(FP), AX
	MOVQ b+8(FP), BX
	MOVQ n+16(FP), CX
	MOVQ CX, R11
	ANDQ $15, R11
	JZ   nomask
	MOVQ $1, R12
	MOVQ R11, CX
	SHLQ CX, R12
	DECQ R12
	KMOVW R12, K1
	MOVQ n+16(FP), CX
nomask:
	ANDQ $-16, CX
	VXORPS X0, X0, X0
	VXORPS X1, X1, X1
	VXORPS X2, X2, X2
	VXORPS X3, X3, X3
	XORQ DI, DI
	LEAQ 64(DI), SI
	CMPQ SI, CX
	JGT  one

quad:
	VMOVUPS (AX)(DI*4), Z4
	VMOVUPS 64(AX)(DI*4), Z5
	VMOVUPS 128(AX)(DI*4), Z6
	VMOVUPS 192(AX)(DI*4), Z7
	VSUBPS (BX)(DI*4), Z4, Z4
	VSUBPS 64(BX)(DI*4), Z5, Z5
	VSUBPS 128(BX)(DI*4), Z6, Z6
	VSUBPS 192(BX)(DI*4), Z7, Z7
	VFMADD231PS Z4, Z4, Z0
	VFMADD231PS Z5, Z5, Z1
	VFMADD231PS Z6, Z6, Z2
	VFMADD231PS Z7, Z7, Z3
	ADDQ $64, DI
	LEAQ 64(DI), SI
	CMPQ SI, CX
	JLE  quad

one:
	CMPQ DI, CX
	JGE  tail
	VMOVUPS (AX)(DI*4), Z4
	VSUBPS (BX)(DI*4), Z4, Z4
	VFMADD231PS Z4, Z4, Z0
	ADDQ $16, DI
	JMP  one

tail:
	TESTQ R11, R11
	JZ    reduce
	VMOVUPS.Z (AX)(DI*4), K1, Z4
	VMOVUPS.Z (BX)(DI*4), K1, Z5
	VSUBPS Z5, Z4, Z4
	VFMADD231PS Z4, Z4, Z0

reduce:
	VADDPS Z1, Z0, Z0
	VADDPS Z3, Z2, Z2
	VADDPS Z2, Z0, Z0
	REDUCE512
	VMOVSS X0, ret+24(FP)
	VZEROUPPER
	RET
