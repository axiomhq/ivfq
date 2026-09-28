#include "textflag.h"

// func rotatePairsAVX2(pairs []pair, block *float32)
// pair is {i, j int32; cs, sn float32}; a block row is 16 float32 (64 bytes).
TEXT ·rotatePairsAVX2(SB), NOSPLIT, $0-32
	MOVQ pairs_base+0(FP), SI
	MOVQ pairs_len+8(FP), CX
	MOVQ block+24(FP), DI
	TESTQ CX, CX
	JZ   done

loop:
	MOVLQSX 0(SI), AX
	MOVLQSX 4(SI), BX
	SHLQ $6, AX
	SHLQ $6, BX
	VBROADCASTSS 8(SI), Y0  // cs
	VBROADCASTSS 12(SI), Y1 // sn
	VMOVUPS 0(DI)(AX*1), Y2  // a, lanes 0-7
	VMOVUPS 32(DI)(AX*1), Y3 // a, lanes 8-15
	VMOVUPS 0(DI)(BX*1), Y4  // b, lanes 0-7
	VMOVUPS 32(DI)(BX*1), Y5 // b, lanes 8-15

	// a' = cs*a - sn*b
	VMULPS Y2, Y0, Y6
	VMULPS Y4, Y1, Y7
	VSUBPS Y7, Y6, Y6
	VMULPS Y3, Y0, Y8
	VMULPS Y5, Y1, Y9
	VSUBPS Y9, Y8, Y8

	// b' = sn*a + cs*b
	VMULPS Y2, Y1, Y10
	VMULPS Y4, Y0, Y11
	VADDPS Y11, Y10, Y10
	VMULPS Y3, Y1, Y12
	VMULPS Y5, Y0, Y13
	VADDPS Y13, Y12, Y12

	VMOVUPS Y6, 0(DI)(AX*1)
	VMOVUPS Y8, 32(DI)(AX*1)
	VMOVUPS Y10, 0(DI)(BX*1)
	VMOVUPS Y12, 32(DI)(BX*1)

	ADDQ $16, SI
	DECQ CX
	JNZ  loop

done:
	VZEROUPPER
	RET
