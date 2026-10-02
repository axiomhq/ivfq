#include "textflag.h"

// func fastScanAVX2(nibs *byte, lut *byte, pairs, blocks int, out *uint16)
TEXT ·fastScanAVX2(SB), NOSPLIT, $0-40
	MOVQ nibs+0(FP), SI
	MOVQ lut+8(FP), DX
	MOVQ pairs+16(FP), CX
	MOVQ blocks+24(FP), R8
	MOVQ out+32(FP), DI
	MOVQ $0x0f0f0f0f0f0f0f0f, AX
	MOVQ AX, X15
	VPBROADCASTQ X15, Y15 // 0x0F per byte
	VPCMPEQW Y14, Y14, Y14
	VPSRLW $8, Y14, Y14   // 0x00FF per word

block:
	VPXOR Y0, Y0, Y0 // low nibbles, even bytes: rows 0, 2, ... 14
	VPXOR Y1, Y1, Y1 // low nibbles, odd bytes: rows 1, 3, ... 15
	VPXOR Y2, Y2, Y2 // high nibbles, even bytes: rows 16, 18, ... 30
	VPXOR Y3, Y3, Y3 // high nibbles, odd bytes: rows 17, 19, ... 31
	MOVQ DX, BX
	MOVQ CX, R9

pair:
	VMOVDQU (SI), Y4
	VMOVDQU (BX), Y5
	VPAND Y15, Y4, Y6
	VPSRLW $4, Y4, Y7
	VPAND Y15, Y7, Y7
	VPSHUFB Y6, Y5, Y6
	VPSHUFB Y7, Y5, Y7
	VPAND Y14, Y6, Y8
	VPADDW Y8, Y0, Y0
	VPSRLW $8, Y6, Y6
	VPADDW Y6, Y1, Y1
	VPAND Y14, Y7, Y8
	VPADDW Y8, Y2, Y2
	VPSRLW $8, Y7, Y7
	VPADDW Y7, Y3, Y3
	ADDQ $32, SI
	ADDQ $32, BX
	DECQ R9
	JNZ  pair

	// Fold the two groups' lanes, then interleave even and odd rows.
	VEXTRACTI128 $1, Y0, X8
	VPADDW X8, X0, X0
	VEXTRACTI128 $1, Y1, X8
	VPADDW X8, X1, X1
	VEXTRACTI128 $1, Y2, X8
	VPADDW X8, X2, X2
	VEXTRACTI128 $1, Y3, X8
	VPADDW X8, X3, X3
	VPUNPCKLWD X1, X0, X8
	VMOVDQU X8, (DI)
	VPUNPCKHWD X1, X0, X8
	VMOVDQU X8, 16(DI)
	VPUNPCKLWD X3, X2, X8
	VMOVDQU X8, 32(DI)
	VPUNPCKHWD X3, X2, X8
	VMOVDQU X8, 48(DI)
	ADDQ $64, DI
	DECQ R8
	JNZ  block

	VZEROUPPER
	RET
