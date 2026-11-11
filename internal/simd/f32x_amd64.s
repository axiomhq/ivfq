#include "textflag.h"

DATA expmask<>+0(SB)/4, $0x7f800000
GLOBL expmask<>(SB), RODATA|NOPTR, $4

// func nonFinite(x *float32, n int) bool
// n is a multiple of 8, n >= 8. True when any value has an all-ones exponent.
TEXT ·nonFinite(SB), NOSPLIT, $0-17
	MOVQ x+0(FP), SI
	MOVQ n+8(FP), CX
	SHRQ $3, CX
	VPBROADCASTD expmask<>(SB), Y1
	VPXOR Y2, Y2, Y2

loop:
	VPAND (SI), Y1, Y0
	VPCMPEQD Y1, Y0, Y0
	VPOR Y0, Y2, Y2
	ADDQ $32, SI
	DECQ CX
	JNZ  loop

	VPTEST Y2, Y2
	SETNE ret+16(FP)
	VZEROUPPER
	RET

// func addF64(dst *float64, src *float32, n int)
// n is a multiple of 4, n >= 4.
TEXT ·addF64(SB), NOSPLIT, $0-24
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ n+16(FP), CX
	SHRQ $2, CX

loop:
	VCVTPS2PD (SI), Y0
	VADDPD (DI), Y0, Y0
	VMOVUPD Y0, (DI)
	ADDQ $16, SI
	ADDQ $32, DI
	DECQ CX
	JNZ  loop

	VZEROUPPER
	RET

// func scaleF64(x *float32, n int, s float64)
// n is a multiple of 4, n >= 4.
TEXT ·scaleF64(SB), NOSPLIT, $0-24
	MOVQ x+0(FP), SI
	MOVQ n+8(FP), CX
	SHRQ $2, CX
	VBROADCASTSD s+16(FP), Y1

loop:
	VCVTPS2PD (SI), Y0
	VMULPD Y1, Y0, Y0
	VCVTPD2PSY Y0, X0
	VMOVUPS X0, (SI)
	ADDQ $16, SI
	DECQ CX
	JNZ  loop

	VZEROUPPER
	RET
