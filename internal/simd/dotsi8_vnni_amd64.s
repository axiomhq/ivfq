#include "textflag.h"

// func dotsI8VNNI(q *int8, block *int8, dims, rows int, out *int32)
// out[r] = sum q[j]*row_r[j], exactly, 64 lanes a step with AVX-512 VNNI.
// VPDPBUSD multiplies unsigned by signed, so q is biased to u = q^0x80 =
// q+128 and the bias taken back out per row: sum q*b = sum u*b - 128*sum b,
// with sum b from the same instruction against a vector of ones. The last
// partial vector is a masked load: masked-off lanes read 0 on both sides.
TEXT ·dotsI8VNNI(SB), NOSPLIT, $0-40
	MOVQ q+0(FP), AX
	MOVQ block+8(FP), BX
	MOVQ dims+16(FP), CX
	MOVQ rows+24(FP), DX
	MOVQ out+32(FP), R9

	MOVQ $0x8080808080808080, R10
	VPBROADCASTQ R10, Z4 // 0x80 in every byte
	MOVQ $0x0101010101010101, R10
	VPBROADCASTQ R10, Z5 // 1 in every byte

	MOVQ CX, R11
	ANDQ $63, R11 // tail lanes
	JZ   nomask
	MOVQ $1, R12
	MOVQ R11, CX
	SHLQ CX, R12
	DECQ R12
	KMOVQ R12, K1
	MOVQ dims+16(FP), CX
nomask:
	MOVQ CX, R8
	ANDQ $-64, R8 // lanes covered by whole vectors

row:
	VPXORD Z0, Z0, Z0 // sum u*b
	VPXORD Z1, Z1, Z1 // sum b
	VPXORD Z2, Z2, Z2
	VPXORD Z3, Z3, Z3
	XORQ DI, DI
	LEAQ 128(DI), SI
	CMPQ SI, R8
	JGT  one

pair:
	VMOVDQU64 (AX)(DI*1), Z6
	VMOVDQU64 64(AX)(DI*1), Z7
	VPXORD Z4, Z6, Z6
	VPXORD Z4, Z7, Z7
	VMOVDQU64 (BX)(DI*1), Z8
	VMOVDQU64 64(BX)(DI*1), Z9
	VPDPBUSD Z8, Z6, Z0
	VPDPBUSD Z9, Z7, Z2
	VPDPBUSD Z8, Z5, Z1
	VPDPBUSD Z9, Z5, Z3
	ADDQ $128, DI
	LEAQ 128(DI), SI
	CMPQ SI, R8
	JLE  pair

one:
	CMPQ DI, R8
	JGE  tail
	VMOVDQU64 (AX)(DI*1), Z6
	VPXORD Z4, Z6, Z6
	VMOVDQU64 (BX)(DI*1), Z8
	VPDPBUSD Z8, Z6, Z0
	VPDPBUSD Z8, Z5, Z1
	ADDQ $64, DI

tail:
	TESTQ R11, R11
	JZ    reduce
	VMOVDQU8.Z (AX)(DI*1), K1, Z6
	VPXORD Z4, Z6, Z6
	VMOVDQU8.Z (BX)(DI*1), K1, Z8
	VPDPBUSD Z8, Z6, Z0
	VPDPBUSD Z8, Z5, Z1

reduce:
	VPADDD Z2, Z0, Z0
	VPADDD Z3, Z1, Z1
	VPSLLD $7, Z1, Z1
	VPSUBD Z1, Z0, Z0
	VEXTRACTI64X4 $1, Z0, Y1
	VPADDD Y1, Y0, Y0
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
