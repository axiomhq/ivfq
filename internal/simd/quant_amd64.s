#include "textflag.h"

DATA half8<>+0(SB)/8, $0.5
GLOBL half8<>(SB), RODATA|NOPTR, $8
DATA one8<>+0(SB)/8, $1.0
GLOBL one8<>(SB), RODATA|NOPTR, $8
DATA absmask8<>+0(SB)/8, $0x7fffffffffffffff
GLOBL absmask8<>(SB), RODATA|NOPTR, $8
// VPSHUFB mask taking the low byte of each int32 lane to bytes 0..3.
DATA pack8<>+0(SB)/4, $0x0c080400
DATA pack8<>+4(SB)/4, $0x80808080
DATA pack8<>+8(SB)/4, $0x80808080
DATA pack8<>+12(SB)/4, $0x80808080
GLOBL pack8<>(SB), RODATA|NOPTR, $16

// func blockSumSq(block *float32, dims int, sums *float64)
TEXT ·blockSumSq(SB), NOSPLIT, $0-24
	MOVQ block+0(FP), SI
	MOVQ dims+8(FP), CX
	MOVQ sums+16(FP), DI
	VMOVUPD (DI), Y4
	VMOVUPD 32(DI), Y5
	VMOVUPD 64(DI), Y6
	VMOVUPD 96(DI), Y7

loop:
	VCVTPS2PD (SI), Y0
	VCVTPS2PD 16(SI), Y1
	VCVTPS2PD 32(SI), Y2
	VCVTPS2PD 48(SI), Y3
	VMULPD Y0, Y0, Y0
	VMULPD Y1, Y1, Y1
	VMULPD Y2, Y2, Y2
	VMULPD Y3, Y3, Y3
	VADDPD Y0, Y4, Y4
	VADDPD Y1, Y5, Y5
	VADDPD Y2, Y6, Y6
	VADDPD Y3, Y7, Y7
	ADDQ $64, SI
	DECQ CX
	JNZ  loop

	VMOVUPD Y4, (DI)
	VMOVUPD Y5, 32(DI)
	VMOVUPD Y6, 64(DI)
	VMOVUPD Y7, 96(DI)
	VZEROUPPER
	RET

// func blockBits(block *float32, n int, abs *float64, out *uint32)
TEXT ·blockBits(SB), NOSPLIT, $0-32
	MOVQ block+0(FP), SI
	MOVQ n+8(FP), CX
	MOVQ abs+16(FP), DI
	MOVQ out+24(FP), R8
	VMOVUPD (DI), Y4
	VMOVUPD 32(DI), Y5
	VMOVUPD 64(DI), Y6
	VMOVUPD 96(DI), Y7
	VBROADCASTSD absmask8<>(SB), Y15
	VPXOR Y14, Y14, Y14 // bits, rows 0-7
	VPXOR Y13, Y13, Y13 // bits, rows 8-15
	VXORPS Y12, Y12, Y12 // zero
	XORQ AX, AX

loop:
	VCVTPS2PD (SI), Y0
	VCVTPS2PD 16(SI), Y1
	VCVTPS2PD 32(SI), Y2
	VCVTPS2PD 48(SI), Y3
	VANDPD Y15, Y0, Y0
	VANDPD Y15, Y1, Y1
	VANDPD Y15, Y2, Y2
	VANDPD Y15, Y3, Y3
	VADDPD Y0, Y4, Y4
	VADDPD Y1, Y5, Y5
	VADDPD Y2, Y6, Y6
	VADDPD Y3, Y7, Y7
	VMOVUPS (SI), Y8
	VMOVUPS 32(SI), Y9
	VCMPPS $0x1d, Y12, Y8, Y8 // >= 0, false for NaN
	VCMPPS $0x1d, Y12, Y9, Y9
	VPSRLD $31, Y8, Y8
	VPSRLD $31, Y9, Y9
	VMOVQ AX, X10
	VPSLLD X10, Y8, Y8
	VPSLLD X10, Y9, Y9
	VPOR Y8, Y14, Y14
	VPOR Y9, Y13, Y13
	ADDQ $64, SI
	INCQ AX
	CMPQ AX, CX
	JLT  loop

	VMOVUPD Y4, (DI)
	VMOVUPD Y5, 32(DI)
	VMOVUPD Y6, 64(DI)
	VMOVUPD Y7, 96(DI)
	VMOVDQU Y14, (R8)
	VMOVDQU Y13, 32(R8)
	VZEROUPPER
	RET

// func minMaxScaled(x *float32, n int, mul float64) (lo, hi float64)
TEXT ·minMaxScaled(SB), NOSPLIT, $0-40
	MOVQ x+0(FP), SI
	MOVQ n+8(FP), CX
	VBROADCASTSD mul+16(FP), Y4
	VCVTPS2PD (SI), Y0
	VMULPD Y4, Y0, Y0
	VMOVAPD Y0, Y1
	ADDQ $16, SI
	SUBQ $4, CX
	JZ   reduce

loop:
	VCVTPS2PD (SI), Y2
	VMULPD Y4, Y2, Y2
	VMINPD Y2, Y0, Y0
	VMAXPD Y2, Y1, Y1
	ADDQ $16, SI
	SUBQ $4, CX
	JNZ  loop

reduce:
	VEXTRACTF128 $1, Y0, X2
	VMINPD X2, X0, X0
	VPERMILPD $1, X0, X2
	VMINSD X2, X0, X0
	VEXTRACTF128 $1, Y1, X3
	VMAXPD X3, X1, X1
	VPERMILPD $1, X1, X3
	VMAXSD X3, X1, X1
	VMOVSD X0, lo+24(FP)
	VMOVSD X1, hi+32(FP)
	VZEROUPPER
	RET

// func sumSq(x *float32, n int) float64
TEXT ·sumSq(SB), NOSPLIT, $0-24
	MOVQ x+0(FP), SI
	MOVQ n+8(FP), CX
	SHRQ $2, CX
	VXORPD Y0, Y0, Y0

loop:
	VCVTPS2PD (SI), Y2
	VMULPD Y2, Y2, Y2
	VADDPD Y2, Y0, Y0
	ADDQ $16, SI
	DECQ CX
	JNZ  loop

	VEXTRACTF128 $1, Y0, X1
	VADDPD X1, X0, X0
	VPERMILPD $1, X0, X1
	VADDSD X1, X0, X0
	VMOVSD X0, ret+16(FP)
	VZEROUPPER
	RET

// func quantCodes(codes *byte, x *float32, n int, mul, lo, delta, levels float64) float64
TEXT ·quantCodes(SB), NOSPLIT, $0-64
	MOVQ codes+0(FP), DI
	MOVQ x+8(FP), SI
	MOVQ n+16(FP), CX
	SHRQ $2, CX
	VBROADCASTSD mul+24(FP), Y4
	VBROADCASTSD lo+32(FP), Y5
	VBROADCASTSD delta+40(FP), Y6
	VBROADCASTSD levels+48(FP), Y7
	VBROADCASTSD half8<>(SB), Y10
	VBROADCASTSD one8<>(SB), Y11
	VXORPD Y12, Y12, Y12
	VXORPD Y0, Y0, Y0
	VMOVDQU pack8<>(SB), X15

loop:
	VCVTPS2PD (SI), Y2
	VMULPD Y4, Y2, Y2     // x
	VSUBPD Y5, Y2, Y3     // x - lo
	VDIVPD Y6, Y3, Y3     // y
	VROUNDPD $3, Y3, Y8   // t = trunc(y)
	VSUBPD Y8, Y3, Y9     // y - t
	VCMPPD $0x1d, Y10, Y9, Y9
	VANDPD Y11, Y9, Y9
	VADDPD Y9, Y8, Y8     // t++ when y-t >= 0.5
	VMAXPD Y12, Y8, Y8
	VMINPD Y7, Y8, Y8
	VMULPD Y6, Y8, Y13    // delta*t
	VADDPD Y13, Y5, Y13   // lo + delta*t
	VSUBPD Y2, Y13, Y13   // - x
	VMULPD Y13, Y13, Y13
	VADDPD Y13, Y0, Y0
	VCVTTPD2DQY Y8, X14
	VPSHUFB X15, X14, X14
	VMOVD X14, (DI)
	ADDQ $16, SI
	ADDQ $4, DI
	DECQ CX
	JNZ  loop

	VEXTRACTF128 $1, Y0, X1
	VADDPD X1, X0, X0
	VPERMILPD $1, X0, X1
	VADDSD X1, X0, X0
	VMOVSD X0, ret+56(FP)
	VZEROUPPER
	RET

DATA sfNegZero<>+0(SB)/8, $0x8000000000000000
GLOBL sfNegZero<>(SB), RODATA|NOPTR, $8
DATA sfOne<>+0(SB)/8, $0x3ff0000000000000
GLOBL sfOne<>(SB), RODATA|NOPTR, $8
DATA sfHalf<>+0(SB)/8, $0x3fe0000000000000
GLOBL sfHalf<>(SB), RODATA|NOPTR, $8
DATA sfNegOne<>+0(SB)/8, $0xbff0000000000000
GLOBL sfNegOne<>(SB), RODATA|NOPTR, $8

// func scoreFast4(n int, sums, ones *uint16, norms, aligns, scores *float32, k *float64, cos int)
// Four rows a step in float64 lanes: the Go loop's operations in its order
// (a doubling is x+x, a halving x*0.5, both exact), with both branches of
// its conditions computed and blended. VMAXPD takes the square second, so
// a zero distance is +0 as Go's max gives.
TEXT ·scoreFast4(SB), NOSPLIT, $0-64
	MOVQ n+0(FP), CX
	MOVQ sums+8(FP), SI
	MOVQ ones+16(FP), R8
	MOVQ norms+24(FP), R9
	MOVQ aligns+32(FP), R10
	MOVQ scores+40(FP), DI
	MOVQ k+48(FP), DX
	MOVQ cos+56(FP), R11
	VBROADCASTSD 0(DX), Y8   // delta
	VBROADCASTSD 8(DX), Y9   // low
	VBROADCASTSD 16(DX), Y10 // tail
	VBROADCASTSD 24(DX), Y11 // scale
	VBROADCASTSD 32(DX), Y12 // qn
	VBROADCASTSD 40(DX), Y13 // q2
	VXORPD Y14, Y14, Y14
	SHRQ $2, CX
	JZ   sfdone

sfloop:
	VPMOVZXWD (SI), X0
	VCVTDQ2PD X0, Y0  // sums
	VPMOVZXWD (R8), X1
	VCVTDQ2PD X1, Y1  // ones
	VCVTPS2PD (R9), Y2  // norm
	VCVTPS2PD (R10), Y3 // align
	VMULPD Y8, Y0, Y0  // delta*sums
	VMULPD Y9, Y1, Y1  // low*ones
	VADDPD Y1, Y0, Y0
	VADDPD Y0, Y0, Y0  // 2*(...)
	VSUBPD Y10, Y0, Y0 // - tail
	VMULPD Y11, Y0, Y0 // * scale
	VDIVPD Y3, Y0, Y0  // / align: ip
	VADDPD Y2, Y2, Y4  // 2*norm
	VMULPD Y12, Y4, Y4 // *qn
	VMULPD Y0, Y4, Y4  // *ip
	VMULPD Y2, Y2, Y5  // norm*norm
	VADDPD Y13, Y5, Y5 // + q2
	VSUBPD Y4, Y5, Y6  // - 2*norm*qn*ip
	VCMPPD $0x1e, Y14, Y2, Y7  // norm > 0
	VCMPPD $0x1e, Y14, Y3, Y15 // align > 0
	VANDPD Y15, Y7, Y7
	VBLENDVPD Y7, Y6, Y5, Y5
	VSUBPD Y12, Y2, Y6 // gap
	VMULPD Y6, Y6, Y6
	VMAXPD Y6, Y5, Y5  // max(dist2, gap*gap)
	TESTQ R11, R11
	JNZ  sfcos
	VBROADCASTSD sfNegZero<>(SB), Y15
	VXORPD Y15, Y5, Y5 // -dist2
	JMP  sfstore

sfcos:
	VBROADCASTSD sfHalf<>(SB), Y15
	VMULPD Y15, Y5, Y5 // dist2/2
	VBROADCASTSD sfOne<>(SB), Y15
	VSUBPD Y5, Y15, Y5 // 1 - dist2/2
	VBROADCASTSD sfNegOne<>(SB), Y15
	VMAXPD Y15, Y5, Y5 // max(-1, x)
	VBROADCASTSD sfOne<>(SB), Y15
	VMINPD Y15, Y5, Y5 // min(1, x)

sfstore:
	VBROADCASTSD sfNegOne<>(SB), Y15
	VCMPPD $0x00, Y15, Y3, Y7 // align == -1: a zero row
	VBLENDVPD Y7, Y14, Y5, Y5
	VCVTPD2PSY Y5, X5
	VMOVUPS X5, (DI)
	ADDQ $8, SI
	ADDQ $8, R8
	ADDQ $16, R9
	ADDQ $16, R10
	ADDQ $16, DI
	DECQ CX
	JNZ  sfloop

sfdone:
	VZEROUPPER
	RET

// func subSumSq(x, c *float32, n int, lanes *[4]float64)
// Eight floats a step: x -= c in float32, stored, then the low four's
// squares and the high four's added to the four lanes in that order, as
// sumSq adds them four at a time.
TEXT ·subSumSq(SB), NOSPLIT, $0-32
	MOVQ x+0(FP), SI
	MOVQ c+8(FP), DX
	MOVQ n+16(FP), CX
	MOVQ lanes+24(FP), DI
	SHRQ $3, CX
	VXORPD Y0, Y0, Y0

sssloop:
	VMOVUPS (SI), Y1
	VSUBPS (DX), Y1, Y1
	VMOVUPS Y1, (SI)
	VCVTPS2PD X1, Y2
	VEXTRACTF128 $1, Y1, X3
	VCVTPS2PD X3, Y3
	VMULPD Y2, Y2, Y2
	VADDPD Y2, Y0, Y0
	VMULPD Y3, Y3, Y3
	VADDPD Y3, Y0, Y0
	ADDQ $32, SI
	ADDQ $32, DX
	DECQ CX
	JNZ  sssloop

	VMOVUPD Y0, (DI)
	VZEROUPPER
	RET

// func planes32(codes *byte, n int, planes *uint64, words int) uint64
// 32 codes a step: shifting each 16-bit word left by 7-p puts bit p of
// both its bytes at their bit 7 (codes are below 16, so nothing of the low
// byte reaches the high byte's bit 7), and VPMOVMSKB gathers them: 32 bits
// of plane p, stored at the step's 4-byte slot. VPSADBW sums the codes.
TEXT ·planes32(SB), NOSPLIT, $0-40
	MOVQ codes+0(FP), SI
	MOVQ n+8(FP), CX
	MOVQ planes+16(FP), DI
	MOVQ words+24(FP), R8
	SHLQ $3, R8 // a plane's bytes
	LEAQ (DI)(R8*1), R9
	LEAQ (R9)(R8*1), R10
	LEAQ (R10)(R8*1), R11
	SHRQ $5, CX
	VPXOR Y5, Y5, Y5
	VPXOR Y6, Y6, Y6

plloop:
	VMOVDQU (SI), Y0
	VPSADBW Y5, Y0, Y1
	VPADDQ Y1, Y6, Y6
	VPSLLW $7, Y0, Y1
	VPMOVMSKB Y1, AX
	MOVL AX, (DI)
	VPSLLW $6, Y0, Y1
	VPMOVMSKB Y1, AX
	MOVL AX, (R9)
	VPSLLW $5, Y0, Y1
	VPMOVMSKB Y1, AX
	MOVL AX, (R10)
	VPSLLW $4, Y0, Y1
	VPMOVMSKB Y1, AX
	MOVL AX, (R11)
	ADDQ $32, SI
	ADDQ $4, DI
	ADDQ $4, R9
	ADDQ $4, R10
	ADDQ $4, R11
	DECQ CX
	JNZ  plloop

	VEXTRACTI128 $1, Y6, X1
	VPADDQ X1, X6, X6
	VPSHUFD $0x4e, X6, X1
	VPADDQ X1, X6, X6
	VMOVQ X6, AX
	MOVQ AX, ret+32(FP)
	VZEROUPPER
	RET

// func planesLUTAVX2(planes *uint64, words int, lut *byte, pairs int, tbl *byte)
// Each byte of each of the 4 planes holds two 4-dim nibbles (low = group 2k,
// high = group 2k+1). tbl[n*16+m] = popcount(n&m) <= 4, so <<2 and <<3 never
// cross byte boundaries and the weighted sum stays <= 60.
TEXT ·planesLUTAVX2(SB), NOSPLIT, $0-40
	MOVQ planes+0(FP), SI
	MOVQ words+8(FP), R8
	SHLQ $3, R8 // a plane's bytes
	LEAQ (SI)(R8*1), R9
	LEAQ (R9)(R8*1), R10
	LEAQ (R10)(R8*1), R11
	MOVQ lut+16(FP), DI
	MOVQ pairs+24(FP), CX
	MOVQ tbl+32(FP), R12

plutloop:
	MOVBLZX (SI), AX
	MOVBLZX (R9), BX
	MOVBLZX (R10), DX
	MOVBLZX (R11), R8

	// Low nibble: offset = (b & 0x0F) << 4
	MOVQ AX, R13
	ANDQ $0x0F, R13
	SHLQ $4, R13
	VMOVDQU (R12)(R13*1), X0
	MOVQ BX, R13
	ANDQ $0x0F, R13
	SHLQ $4, R13
	VMOVDQU (R12)(R13*1), X1
	VPADDB X1, X1, X1
	VPADDB X1, X0, X0
	MOVQ DX, R13
	ANDQ $0x0F, R13
	SHLQ $4, R13
	VMOVDQU (R12)(R13*1), X2
	VPSLLW $2, X2, X2
	VPADDB X2, X0, X0
	MOVQ R8, R13
	ANDQ $0x0F, R13
	SHLQ $4, R13
	VMOVDQU (R12)(R13*1), X3
	VPSLLW $3, X3, X3
	VPADDB X3, X0, X0
	VMOVDQU X0, (DI)

	// High nibble: offset = b & 0xF0
	ANDQ $0xF0, AX
	VMOVDQU (R12)(AX*1), X0
	ANDQ $0xF0, BX
	VMOVDQU (R12)(BX*1), X1
	VPADDB X1, X1, X1
	VPADDB X1, X0, X0
	ANDQ $0xF0, DX
	VMOVDQU (R12)(DX*1), X2
	VPSLLW $2, X2, X2
	VPADDB X2, X0, X0
	ANDQ $0xF0, R8
	VMOVDQU (R12)(R8*1), X3
	VPSLLW $3, X3, X3
	VPADDB X3, X0, X0
	VMOVDQU X0, 16(DI)

	INCQ SI
	INCQ R9
	INCQ R10
	INCQ R11
	ADDQ $32, DI
	DECQ CX
	JNZ  plutloop

	VZEROUPPER
	RET

DATA unpackBitMask<>+0(SB)/4, $1
DATA unpackBitMask<>+4(SB)/4, $2
DATA unpackBitMask<>+8(SB)/4, $4
DATA unpackBitMask<>+12(SB)/4, $8
DATA unpackBitMask<>+16(SB)/4, $16
DATA unpackBitMask<>+20(SB)/4, $32
DATA unpackBitMask<>+24(SB)/4, $64
DATA unpackBitMask<>+28(SB)/4, $128
GLOBL unpackBitMask<>(SB), RODATA|NOPTR, $32

DATA unpackSignBit<>+0(SB)/4, $0x80000000
GLOBL unpackSignBit<>(SB), RODATA|NOPTR, $4

// func unpack1BitAVX2(dst, base *float32, words *uint64, n int, s float32)
// Eight floats per byte of words: dst[j] = base[j] + (bit ? +s : -s).
TEXT ·unpack1BitAVX2(SB), NOSPLIT, $0-36
	MOVQ dst+0(FP), DI
	MOVQ base+8(FP), SI
	MOVQ words+16(FP), BX
	MOVQ n+24(FP), CX
	SHRQ $3, CX
	VBROADCASTSS s+32(FP), Y4
	VPBROADCASTD unpackSignBit<>(SB), Y5
	VMOVDQU unpackBitMask<>(SB), Y6
	VPXOR Y7, Y7, Y7

u1loop:
	MOVBLZX (BX), AX
	VMOVD AX, X0
	VPBROADCASTD X0, Y0
	VPAND Y6, Y0, Y0
	VPCMPEQD Y7, Y0, Y0
	VPAND Y5, Y0, Y0
	VXORPS Y4, Y0, Y0
	VADDPS (SI), Y0, Y0
	VMOVUPS Y0, (DI)
	INCQ BX
	ADDQ $32, SI
	ADDQ $32, DI
	DECQ CX
	JNZ  u1loop

	VZEROUPPER
	RET

// func subStoreSumSq(dst, a, b *float32, n int, lanes *[4]float64)
// Eight floats a step: dst = a - b in float32, then 4-lane float64 SumSq.
TEXT ·subStoreSumSq(SB), NOSPLIT, $0-40
	MOVQ dst+0(FP), R8
	MOVQ a+8(FP), SI
	MOVQ b+16(FP), DX
	MOVQ n+24(FP), CX
	MOVQ lanes+32(FP), DI
	SHRQ $3, CX
	VXORPD Y0, Y0, Y0

ssssloop:
	VMOVUPS (SI), Y1
	VSUBPS (DX), Y1, Y1
	VMOVUPS Y1, (R8)
	VCVTPS2PD X1, Y2
	VEXTRACTF128 $1, Y1, X3
	VCVTPS2PD X3, Y3
	VMULPD Y2, Y2, Y2
	VADDPD Y2, Y0, Y0
	VMULPD Y3, Y3, Y3
	VADDPD Y3, Y0, Y0
	ADDQ $32, SI
	ADDQ $32, DX
	ADDQ $32, R8
	DECQ CX
	JNZ  ssssloop

	VMOVUPD Y0, (DI)
	VZEROUPPER
	RET

DATA signAbsMask<>+0(SB)/4, $0x7fffffff
GLOBL signAbsMask<>(SB), RODATA|NOPTR, $4

// func signAbsAVX2(x *float32, n int, words *uint64, lanes *[4]float64)
// Eight floats a step: pack (x >= 0) bits into words and accumulate |x| in
// four float64 lanes.
TEXT ·signAbsAVX2(SB), NOSPLIT, $0-32
	MOVQ x+0(FP), SI
	MOVQ n+8(FP), CX
	MOVQ words+16(FP), DI
	MOVQ lanes+24(FP), DX
	SHRQ $3, CX
	VXORPD Y0, Y0, Y0
	VPBROADCASTD signAbsMask<>(SB), Y6
	VXORPS Y7, Y7, Y7

saloop:
	VMOVUPS (SI), Y1
	VCMPPS $0x1d, Y7, Y1, Y2 // x >= 0.0 (CMP_GE_OQ)
	VMOVMSKPS Y2, AX
	MOVB AL, (DI)
	VANDPS Y6, Y1, Y1
	VCVTPS2PD X1, Y2
	VADDPD Y2, Y0, Y0
	VEXTRACTF128 $1, Y1, X3
	VCVTPS2PD X3, Y3
	VADDPD Y3, Y0, Y0
	ADDQ $32, SI
	INCQ DI
	DECQ CX
	JNZ  saloop

	VMOVUPD Y0, (DX)
	VZEROUPPER
	RET

// VPSHUFB masks for CodesLUT: each 16-byte lane selects code b where bit b of
// entry index m is 1, and 0 (0x80) where bit b is 0.
DATA clMask0<>+0(SB)/8, $0x0080008000800080
DATA clMask0<>+8(SB)/8, $0x0080008000800080
GLOBL clMask0<>(SB), RODATA|NOPTR, $16

DATA clMask1<>+0(SB)/8, $0x0101808001018080
DATA clMask1<>+8(SB)/8, $0x0101808001018080
GLOBL clMask1<>(SB), RODATA|NOPTR, $16

DATA clMask2<>+0(SB)/8, $0x0202020280808080
DATA clMask2<>+8(SB)/8, $0x0202020280808080
GLOBL clMask2<>(SB), RODATA|NOPTR, $16

DATA clMask3<>+0(SB)/8, $0x8080808080808080
DATA clMask3<>+8(SB)/8, $0x0303030303030303
GLOBL clMask3<>(SB), RODATA|NOPTR, $16

// func codesLUTAVX2(codes, lut *byte, pairs int)
// Eight codes (two 4-dim groups) -> 32 bytes of FastScan LUT per iteration.
TEXT ·codesLUTAVX2(SB), NOSPLIT, $0-24
	MOVQ codes+0(FP), SI
	MOVQ lut+8(FP), DI
	MOVQ pairs+16(FP), CX
	VBROADCASTI128 clMask0<>(SB), Y12
	VBROADCASTI128 clMask1<>(SB), Y13
	VBROADCASTI128 clMask2<>(SB), Y14
	VBROADCASTI128 clMask3<>(SB), Y15

clloop:
	VPBROADCASTD (SI), X0
	VPBROADCASTD 4(SI), X1
	VINSERTI128 $1, X1, Y0, Y0
	VPSHUFB Y12, Y0, Y1
	VPSHUFB Y13, Y0, Y2
	VPSHUFB Y14, Y0, Y3
	VPSHUFB Y15, Y0, Y4
	VPADDB Y2, Y1, Y1
	VPADDB Y4, Y3, Y3
	VPADDB Y3, Y1, Y1
	VMOVDQU Y1, (DI)
	ADDQ $8, SI
	ADDQ $32, DI
	DECQ CX
	JNZ  clloop

	VZEROUPPER
	RET
