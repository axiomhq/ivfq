#include "textflag.h"

// func decodeI8AVX2(dst *float32, src *byte, n int)
// dst[i] = float32(int8(src[i])) for i < n, n a multiple of 8.
TEXT ·decodeI8AVX2(SB), NOSPLIT, $0-24
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ n+16(FP), CX
	SHRQ $3, CX
	JZ   done

loop:
	VPMOVSXBD (SI), Y0
	VCVTDQ2PS Y0, Y0
	VMOVUPS   Y0, (DI)
	ADDQ $8, SI
	ADDQ $32, DI
	DECQ CX
	JNZ  loop

done:
	VZEROUPPER
	RET

// func encodeI8AVX2(dst *byte, src *float32, n int) int
// Encodes src eight values at a time while all eight are integers
// (truncation converts them back exactly), saturating to int8, and returns
// how many values it encoded: a multiple of 8, at most n.
TEXT ·encodeI8AVX2(SB), NOSPLIT, $0-32
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ n+16(FP), CX
	SHRQ $3, CX
	XORQ AX, AX
	TESTQ CX, CX
	JZ   done

loop:
	VMOVUPS    (SI), Y0
	VCVTTPS2DQ Y0, Y1
	VCVTDQ2PS  Y1, Y2
	VCMPPS     $0, Y0, Y2, Y3 // EQ_OQ: false for NaN
	VMOVMSKPS  Y3, DX
	CMPL       DX, $0xff
	JNE        done
	VEXTRACTI128 $1, Y1, X4
	VPACKSSDW  X4, X1, X1
	VPACKSSWB  X1, X1, X1
	MOVQ       X1, (DI)
	ADDQ $32, SI
	ADDQ $8, DI
	ADDQ $8, AX
	DECQ CX
	JNZ  loop

done:
	MOVQ AX, ret+24(FP)
	VZEROUPPER
	RET
