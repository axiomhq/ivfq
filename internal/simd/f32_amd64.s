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
// Four 16-lane accumulators. See L2Sq for the lane and reduction rule.
TEXT ·l2sqF32AVX512(SB), NOSPLIT, $0-28
	MOVQ a+0(FP), AX
	MOVQ b+8(FP), BX
	MOVQ n+16(FP), CX
	MOVQ CX, R11
	ANDQ $63, R11
	ANDQ $-64, CX
	VXORPS X0, X0, X0
	VXORPS X1, X1, X1
	VXORPS X2, X2, X2
	VXORPS X3, X3, X3
	XORQ DI, DI
	TESTQ CX, CX
	JZ tail512

quad512:
	VMOVUPS (AX)(DI*4), Z4
	VMOVUPS 64(AX)(DI*4), Z5
	VMOVUPS 128(AX)(DI*4), Z6
	VMOVUPS 192(AX)(DI*4), Z7
	VSUBPS (BX)(DI*4), Z4, Z4
	VSUBPS 64(BX)(DI*4), Z5, Z5
	VSUBPS 128(BX)(DI*4), Z6, Z6
	VSUBPS 192(BX)(DI*4), Z7, Z7
	VMULPS Z4, Z4, Z4
	VMULPS Z5, Z5, Z5
	VMULPS Z6, Z6, Z6
	VMULPS Z7, Z7, Z7
	VADDPS Z4, Z0, Z0
	VADDPS Z5, Z1, Z1
	VADDPS Z6, Z2, Z2
	VADDPS Z7, Z3, Z3
	ADDQ $64, DI
	CMPQ DI, CX
	JLT quad512

tail512:
	TESTQ R11, R11
	JZ reduce512
	MOVQ $1, R12
	MOVQ R11, CX
	SHLQ CX, R12
	DECQ R12
// Consume 16 mask bits per quarter, never remapping a tail to lane zero.
#define L2TAIL512(off, acc) \
	KMOVW R12, K1 \
	VMOVUPS.Z off(AX)(DI*4), K1, Z4 \
	VMOVUPS.Z off(BX)(DI*4), K1, Z5 \
	VSUBPS Z5, Z4, Z4 \
	VMULPS Z4, Z4, Z4 \
	VADDPS Z4, acc, acc \
	SHRQ $16, R12
	L2TAIL512(0, Z0)
	L2TAIL512(64, Z1)
	L2TAIL512(128, Z2)
	L2TAIL512(192, Z3)

reduce512:
	VADDPS Z1, Z0, Z0
	VADDPS Z3, Z2, Z2
	VADDPS Z2, Z0, Z0
	REDUCE512
	VMOVSS X0, ret+24(FP)
	VZEROUPPER
	RET

DATA l2lane<>+0(SB)/4, $0
DATA l2lane<>+4(SB)/4, $1
DATA l2lane<>+8(SB)/4, $2
DATA l2lane<>+12(SB)/4, $3
DATA l2lane<>+16(SB)/4, $4
DATA l2lane<>+20(SB)/4, $5
DATA l2lane<>+24(SB)/4, $6
DATA l2lane<>+28(SB)/4, $7
GLOBL l2lane<>(SB), RODATA|NOPTR, $32

// func l2sqF32AVX2(a *float32, b *float32, n int) float32
// Eight 8-lane accumulators with exactly the same assignment as AVX-512.
TEXT ·l2sqF32AVX2(SB), NOSPLIT, $0-28
	MOVQ a+0(FP), AX
	MOVQ b+8(FP), BX
	MOVQ n+16(FP), CX
	MOVQ CX, R11
	ANDQ $63, R11
	ANDQ $-64, CX
	VXORPS X0, X0, X0
	VXORPS X1, X1, X1
	VXORPS X2, X2, X2
	VXORPS X3, X3, X3
	VXORPS X4, X4, X4
	VXORPS X5, X5, X5
	VXORPS X6, X6, X6
	VXORPS X7, X7, X7
	XORQ DI, DI
	TESTQ CX, CX
	JZ tail256

#define L2VEC256(off, acc) \
	VMOVUPS off(AX)(DI*4), Y8 \
	VSUBPS off(BX)(DI*4), Y8, Y8 \
	VMULPS Y8, Y8, Y8 \
	VADDPS Y8, acc, acc
oct256:
	L2VEC256(0, Y0)
	L2VEC256(32, Y1)
	L2VEC256(64, Y2)
	L2VEC256(96, Y3)
	L2VEC256(128, Y4)
	L2VEC256(160, Y5)
	L2VEC256(192, Y6)
	L2VEC256(224, Y7)
	ADDQ $64, DI
	CMPQ DI, CX
	JLT oct256

tail256:
	TESTQ R11, R11
	JZ reduce256
	VMOVD R11, X9
	VPBROADCASTD X9, Y9
	VMOVDQU l2lane<>(SB), Y10
	MOVQ $8, R12
	VMOVD R12, X11
	VPBROADCASTD X11, Y11
// Valid lanes have remaining > lane; a nonpositive remainder masks all.
#define L2TAIL256(off, acc) \
	VPCMPGTD Y10, Y9, Y12 \
	VMASKMOVPS off(AX)(DI*4), Y12, Y8 \
	VMASKMOVPS off(BX)(DI*4), Y12, Y13 \
	VSUBPS Y13, Y8, Y8 \
	VMULPS Y8, Y8, Y8 \
	VADDPS Y8, acc, acc \
	VPSUBD Y11, Y9, Y9
	L2TAIL256(0, Y0)
	L2TAIL256(32, Y1)
	L2TAIL256(64, Y2)
	L2TAIL256(96, Y3)
	L2TAIL256(128, Y4)
	L2TAIL256(160, Y5)
	L2TAIL256(192, Y6)
	L2TAIL256(224, Y7)

reduce256:
	VADDPS Y2, Y0, Y0
	VADDPS Y6, Y4, Y4
	VADDPS Y4, Y0, Y0
	VADDPS Y3, Y1, Y1
	VADDPS Y7, Y5, Y5
	VADDPS Y5, Y1, Y1
	VADDPS Y1, Y0, Y0
	VEXTRACTF128 $1, Y0, X1
	VADDPS X1, X0, X0
	VPSHUFD $0x4e, X0, X1
	VADDPS X1, X0, X0
	VPSHUFD $0xb1, X0, X1
	VADDPS X1, X0, X0
	VMOVSS X0, ret+24(FP)
	VZEROUPPER
	RET
