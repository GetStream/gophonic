TEXT github.com/GetStream/gophonic/internal/whispergemm.kernel4x16(SB) /Users/thesyncim/Documents/ChatGPT/goinfer/internal/whispergemm/kernel_arm64.go
  kernel_arm64.go:35	0x100180da0		f9400b90		MOVD 16(R28), R16
  kernel_arm64.go:35	0x100180da4		eb3063ff		CMP R16, RSP
  kernel_arm64.go:35	0x100180da8		54002409		BLS 288(PC)
  kernel_arm64.go:35	0x100180dac		f81b0ffe		MOVD.W R30, -80(RSP)
  kernel_arm64.go:35	0x100180db0		f81f83fd		MOVD R29, -8(RSP)
  kernel_arm64.go:35	0x100180db4		d10023fd		SUB $8, RSP, R29
  kernel_arm64.go:35	0x100180db8		f9005fe0		MOVD R0, 184(RSP)
  kernel_arm64.go:35	0x100180dbc		f9006be3		MOVD R3, 208(RSP)
  kernel_arm64.go:35	0x100180dc0		f90077e6		MOVD R6, 232(RSP)
  kernel_arm64.go:35	0x100180dc4		f90083e9		MOVD R9, 256(RSP)
  kernel_arm64.go:35	0x100180dc8		f9008fec		MOVD R12, 280(RSP)
  kernel_arm64.go:37	0x100180dcc		eb05003f		CMP R5, R1
  kernel_arm64.go:37	0x100180dd0		54002288		BHI 276(PC)
  kernel_arm64.go:37	0x100180dd4		eb08003f		CMP R8, R1
  kernel_arm64.go:37	0x100180dd8		54002228		BHI 273(PC)
  kernel_arm64.go:37	0x100180ddc		eb0b003f		CMP R11, R1
  kernel_arm64.go:37	0x100180de0		540021c8		BHI 270(PC)
  kernel_arm64.go:38	0x100180de4		d37cec24		LSL $4, R1, R4
  kernel_arm64.go:38	0x100180de8		eb0111df		CMP R1<<4, R14
  kernel_arm64.go:38	0x100180dec		54002143		BCC 266(PC)
  kernel_arm64.go:43	0x100180df0		a900ffff		STP (ZR, ZR), 8(RSP)
  kernel_arm64.go:43	0x100180df4		b24003e5		ORR $1, ZR, R5
  kernel_arm64.go:43	0x100180df8		390027e5		MOVB R5, 9(RSP)
  kernel_arm64.go:43	0x100180dfc		d2806047		MOVD $770, R7
  kernel_arm64.go:43	0x100180e00		790017e7		MOVH R7, 10(RSP)
  kernel_arm64.go:43	0x100180e04		390037e5		MOVB R5, 13(RSP)
  kernel_arm64.go:43	0x100180e08		79001fe7		MOVH R7, 14(RSP)
  kernel_arm64.go:43	0x100180e0c		d2802005		MOVD $256, R5
  kernel_arm64.go:43	0x100180e10		f2a06045		MOVK $(770<<16), R5
  kernel_arm64.go:43	0x100180e14		f2c02005		MOVK $(256<<32), R5
  kernel_arm64.go:43	0x100180e18		f2e06045		MOVK $(770<<48), R5
  kernel_arm64.go:43	0x100180e1c		f9000be5		MOVD R5, 16(RSP)
  kernel_arm64.go:39	0x100180e20		4f00e400		VMOVI $0, V0.B16
  kernel_arm64.go:43	0x100180e24		3cc083e1		FMOVQ 8(RSP), F1
  kernel_arm64.go:44	0x100180e28		d280a085		MOVD $1284, R5
  kernel_arm64.go:44	0x100180e2c		f2a0e0c5		MOVK $(1798<<16), R5
  kernel_arm64.go:44	0x100180e30		f2c0a085		MOVK $(1284<<32), R5
  kernel_arm64.go:44	0x100180e34		f2e0e0c5		MOVK $(1798<<48), R5
  kernel_arm64.go:44	0x100180e38		a90397e5		STP (R5, R5), 56(RSP)
  kernel_arm64.go:44	0x100180e3c		3cc383e2		FMOVQ 56(RSP), F2
  kernel_arm64.go:45	0x100180e40		d2812105		MOVD $2312, R5
  kernel_arm64.go:45	0x100180e44		f2a16145		MOVK $(2826<<16), R5
  kernel_arm64.go:45	0x100180e48		f2c12105		MOVK $(2312<<32), R5
  kernel_arm64.go:45	0x100180e4c		f2e16145		MOVK $(2826<<48), R5
  kernel_arm64.go:45	0x100180e50		a90297e5		STP (R5, R5), 40(RSP)
  kernel_arm64.go:45	0x100180e54		3cc283e3		FMOVQ 40(RSP), F3
  kernel_arm64.go:46	0x100180e58		d281a185		MOVD $3340, R5
  kernel_arm64.go:46	0x100180e5c		f2a1e1c5		MOVK $(3854<<16), R5
  kernel_arm64.go:46	0x100180e60		f2c1a185		MOVK $(3340<<32), R5
  kernel_arm64.go:46	0x100180e64		f2e1e1c5		MOVK $(3854<<48), R5
  kernel_arm64.go:46	0x100180e68		a90197e5		STP (R5, R5), 24(RSP)
  kernel_arm64.go:46	0x100180e6c		3cc183e4		FMOVQ 24(RSP), F4
  kernel_arm64.go:48	0x100180e70		aa1f03e5		MOVD ZR, R5
  kernel_arm64.go:39	0x100180e74		4ea01c05		VMOV V0.B16, V5.B16
  kernel_arm64.go:48	0x100180e78		4ea51ca6		VMOV V5.B16, V6.B16
  kernel_arm64.go:48	0x100180e7c		4ea61cc7		VMOV V6.B16, V7.B16
  kernel_arm64.go:48	0x100180e80		4ea71ce8		VMOV V7.B16, V8.B16
  kernel_arm64.go:48	0x100180e84		4ea81d09		VMOV V8.B16, V9.B16
  kernel_arm64.go:48	0x100180e88		4ea91d2a		VMOV V9.B16, V10.B16
  kernel_arm64.go:48	0x100180e8c		4eaa1d4b		VMOV V10.B16, V11.B16
  kernel_arm64.go:48	0x100180e90		4eab1d6c		VMOV V11.B16, V12.B16
  kernel_arm64.go:48	0x100180e94		4eac1d8d		VMOV V12.B16, V13.B16
  kernel_arm64.go:48	0x100180e98		4ead1dae		VMOV V13.B16, V14.B16
  kernel_arm64.go:48	0x100180e9c		4eae1dcf		VMOV V14.B16, V15.B16
  kernel_arm64.go:48	0x100180ea0		4eaf1df0		VMOV V15.B16, V16.B16
  kernel_arm64.go:48	0x100180ea4		4eb01e11		VMOV V16.B16, V17.B16
  kernel_arm64.go:48	0x100180ea8		4eb11e32		VMOV V17.B16, V18.B16
  kernel_arm64.go:48	0x100180eac		4eb21e53		VMOV V18.B16, V19.B16
  kernel_arm64.go:48	0x100180eb0		4eb31e74		VMOV V19.B16, V20.B16
  kernel_arm64.go:48	0x100180eb4		14000067		JMP 103(PC)
  kernel_arm64.go:49	0x100180eb8		3dc00175		FMOVQ (R11), F21
  kernel_arm64.go:50	0x100180ebc		3dc001b6		FMOVQ (R13), F22
  kernel_arm64.go:51	0x100180ec0		3dc001d7		FMOVQ (R14), F23
  kernel_arm64.go:52	0x100180ec4		3dc001f8		FMOVQ (R15), F24
  kernel_arm64.go:53	0x100180ec8		8b051988		ADD R5<<6, R12, R8
  kernel_arm64.go:55	0x100180ecc		3dc00119		FMOVQ (R8), F25
  kernel_arm64.go:56	0x100180ed0		3dc0051a		FMOVQ 16(R8), F26
  kernel_arm64.go:57	0x100180ed4		3dc0091b		FMOVQ 32(R8), F27
  kernel_arm64.go:58	0x100180ed8		3dc00d1c		FMOVQ 48(R8), F28
  kernel_arm64.go:59	0x100180edc		4e0102bd		VTBL V1.B16, [V21.B16], V29.B16
  kernel_arm64.go:60	0x100180ee0		4e3dcf34		VFMLA V29.S4, V25.S4, V20.S4
  kernel_arm64.go:61	0x100180ee4		4e3dcf53		VFMLA V29.S4, V26.S4, V19.S4
  kernel_arm64.go:62	0x100180ee8		4e3dcf72		VFMLA V29.S4, V27.S4, V18.S4
  kernel_arm64.go:63	0x100180eec		4e3dcf91		VFMLA V29.S4, V28.S4, V17.S4
  kernel_arm64.go:64	0x100180ef0		4e0102dd		VTBL V1.B16, [V22.B16], V29.B16
  kernel_arm64.go:65	0x100180ef4		4e3dcf30		VFMLA V29.S4, V25.S4, V16.S4
  kernel_arm64.go:66	0x100180ef8		4e3dcf4f		VFMLA V29.S4, V26.S4, V15.S4
  kernel_arm64.go:67	0x100180efc		4e3dcf6e		VFMLA V29.S4, V27.S4, V14.S4
  kernel_arm64.go:68	0x100180f00		4e3dcf8d		VFMLA V29.S4, V28.S4, V13.S4
  kernel_arm64.go:69	0x100180f04		4e0102fd		VTBL V1.B16, [V23.B16], V29.B16
  kernel_arm64.go:70	0x100180f08		4e3dcf2c		VFMLA V29.S4, V25.S4, V12.S4
  kernel_arm64.go:71	0x100180f0c		4e3dcf4b		VFMLA V29.S4, V26.S4, V11.S4
  kernel_arm64.go:72	0x100180f10		4e3dcf6a		VFMLA V29.S4, V27.S4, V10.S4
  kernel_arm64.go:73	0x100180f14		4e3dcf89		VFMLA V29.S4, V28.S4, V9.S4
  kernel_arm64.go:74	0x100180f18		4e01031d		VTBL V1.B16, [V24.B16], V29.B16
  kernel_arm64.go:75	0x100180f1c		4e3dcf28		VFMLA V29.S4, V25.S4, V8.S4
  kernel_arm64.go:76	0x100180f20		4e3dcf47		VFMLA V29.S4, V26.S4, V7.S4
  kernel_arm64.go:77	0x100180f24		4e3dcf66		VFMLA V29.S4, V27.S4, V6.S4
  kernel_arm64.go:78	0x100180f28		4e3dcf80		VFMLA V29.S4, V28.S4, V0.S4
  kernel_arm64.go:81	0x100180f2c		3dc01119		FMOVQ 64(R8), F25
  kernel_arm64.go:82	0x100180f30		3dc0151a		FMOVQ 80(R8), F26
  kernel_arm64.go:83	0x100180f34		3dc0191b		FMOVQ 96(R8), F27
  kernel_arm64.go:84	0x100180f38		3dc01d1c		FMOVQ 112(R8), F28
  kernel_arm64.go:85	0x100180f3c		4e0202bd		VTBL V2.B16, [V21.B16], V29.B16
  kernel_arm64.go:86	0x100180f40		4e3dcf34		VFMLA V29.S4, V25.S4, V20.S4
  kernel_arm64.go:87	0x100180f44		4e3dcf53		VFMLA V29.S4, V26.S4, V19.S4
  kernel_arm64.go:88	0x100180f48		4e3dcf72		VFMLA V29.S4, V27.S4, V18.S4
  kernel_arm64.go:89	0x100180f4c		4e3dcf91		VFMLA V29.S4, V28.S4, V17.S4
  kernel_arm64.go:90	0x100180f50		4e0202dd		VTBL V2.B16, [V22.B16], V29.B16
  kernel_arm64.go:91	0x100180f54		4e3dcf30		VFMLA V29.S4, V25.S4, V16.S4
  kernel_arm64.go:92	0x100180f58		4e3dcf4f		VFMLA V29.S4, V26.S4, V15.S4
  kernel_arm64.go:93	0x100180f5c		4e3dcf6e		VFMLA V29.S4, V27.S4, V14.S4
  kernel_arm64.go:94	0x100180f60		4e3dcf8d		VFMLA V29.S4, V28.S4, V13.S4
  kernel_arm64.go:95	0x100180f64		4e0202fd		VTBL V2.B16, [V23.B16], V29.B16
  kernel_arm64.go:96	0x100180f68		4e3dcf2c		VFMLA V29.S4, V25.S4, V12.S4
  kernel_arm64.go:97	0x100180f6c		4e3dcf4b		VFMLA V29.S4, V26.S4, V11.S4
  kernel_arm64.go:98	0x100180f70		4e3dcf6a		VFMLA V29.S4, V27.S4, V10.S4
  kernel_arm64.go:99	0x100180f74		4e3dcf89		VFMLA V29.S4, V28.S4, V9.S4
  kernel_arm64.go:100	0x100180f78		4e02031d		VTBL V2.B16, [V24.B16], V29.B16
  kernel_arm64.go:101	0x100180f7c		4e3dcf28		VFMLA V29.S4, V25.S4, V8.S4
  kernel_arm64.go:102	0x100180f80		4e3dcf47		VFMLA V29.S4, V26.S4, V7.S4
  kernel_arm64.go:103	0x100180f84		4e3dcf66		VFMLA V29.S4, V27.S4, V6.S4
  kernel_arm64.go:104	0x100180f88		4e3dcf80		VFMLA V29.S4, V28.S4, V0.S4
  kernel_arm64.go:107	0x100180f8c		3dc02119		FMOVQ 128(R8), F25
  kernel_arm64.go:108	0x100180f90		3dc0251a		FMOVQ 144(R8), F26
  kernel_arm64.go:109	0x100180f94		3dc0291b		FMOVQ 160(R8), F27
  kernel_arm64.go:110	0x100180f98		3dc02d1c		FMOVQ 176(R8), F28
  kernel_arm64.go:111	0x100180f9c		4e0302bd		VTBL V3.B16, [V21.B16], V29.B16
  kernel_arm64.go:112	0x100180fa0		4e3dcf34		VFMLA V29.S4, V25.S4, V20.S4
  kernel_arm64.go:113	0x100180fa4		4e3dcf53		VFMLA V29.S4, V26.S4, V19.S4
  kernel_arm64.go:114	0x100180fa8		4e3dcf72		VFMLA V29.S4, V27.S4, V18.S4
  kernel_arm64.go:115	0x100180fac		4e3dcf91		VFMLA V29.S4, V28.S4, V17.S4
  kernel_arm64.go:116	0x100180fb0		4e0302dd		VTBL V3.B16, [V22.B16], V29.B16
  kernel_arm64.go:117	0x100180fb4		4e3dcf30		VFMLA V29.S4, V25.S4, V16.S4
  kernel_arm64.go:118	0x100180fb8		4e3dcf4f		VFMLA V29.S4, V26.S4, V15.S4
  kernel_arm64.go:119	0x100180fbc		4e3dcf6e		VFMLA V29.S4, V27.S4, V14.S4
  kernel_arm64.go:120	0x100180fc0		4e3dcf8d		VFMLA V29.S4, V28.S4, V13.S4
  kernel_arm64.go:121	0x100180fc4		4e0302fd		VTBL V3.B16, [V23.B16], V29.B16
  kernel_arm64.go:122	0x100180fc8		4e3dcf2c		VFMLA V29.S4, V25.S4, V12.S4
  kernel_arm64.go:123	0x100180fcc		4e3dcf4b		VFMLA V29.S4, V26.S4, V11.S4
  kernel_arm64.go:124	0x100180fd0		4e3dcf6a		VFMLA V29.S4, V27.S4, V10.S4
  kernel_arm64.go:125	0x100180fd4		4e3dcf89		VFMLA V29.S4, V28.S4, V9.S4
  kernel_arm64.go:126	0x100180fd8		4e03031d		VTBL V3.B16, [V24.B16], V29.B16
  kernel_arm64.go:127	0x100180fdc		4e3dcf28		VFMLA V29.S4, V25.S4, V8.S4
  kernel_arm64.go:128	0x100180fe0		4e3dcf47		VFMLA V29.S4, V26.S4, V7.S4
  kernel_arm64.go:129	0x100180fe4		4e3dcf66		VFMLA V29.S4, V27.S4, V6.S4
  kernel_arm64.go:130	0x100180fe8		4e3dcf80		VFMLA V29.S4, V28.S4, V0.S4
  kernel_arm64.go:133	0x100180fec		3dc03119		FMOVQ 192(R8), F25
  kernel_arm64.go:134	0x100180ff0		3dc0351a		FMOVQ 208(R8), F26
  kernel_arm64.go:135	0x100180ff4		3dc0391b		FMOVQ 224(R8), F27
  kernel_arm64.go:136	0x100180ff8		3dc03d1c		FMOVQ 240(R8), F28
  kernel_arm64.go:137	0x100180ffc		4e0402b5		VTBL V4.B16, [V21.B16], V21.B16
  kernel_arm64.go:142	0x100181000		4e0402d6		VTBL V4.B16, [V22.B16], V22.B16
  kernel_arm64.go:147	0x100181004		4e0402f7		VTBL V4.B16, [V23.B16], V23.B16
  kernel_arm64.go:152	0x100181008		4e040318		VTBL V4.B16, [V24.B16], V24.B16
  kernel_arm64.go:138	0x10018100c		4e35cf34		VFMLA V21.S4, V25.S4, V20.S4
  kernel_arm64.go:139	0x100181010		4e35cf53		VFMLA V21.S4, V26.S4, V19.S4
  kernel_arm64.go:140	0x100181014		4e35cf72		VFMLA V21.S4, V27.S4, V18.S4
  kernel_arm64.go:141	0x100181018		4e35cf91		VFMLA V21.S4, V28.S4, V17.S4
  kernel_arm64.go:143	0x10018101c		4e36cf30		VFMLA V22.S4, V25.S4, V16.S4
  kernel_arm64.go:144	0x100181020		4e36cf4f		VFMLA V22.S4, V26.S4, V15.S4
  kernel_arm64.go:145	0x100181024		4e36cf6e		VFMLA V22.S4, V27.S4, V14.S4
  kernel_arm64.go:146	0x100181028		4e36cf8d		VFMLA V22.S4, V28.S4, V13.S4
  kernel_arm64.go:148	0x10018102c		4e37cf2c		VFMLA V23.S4, V25.S4, V12.S4
  kernel_arm64.go:149	0x100181030		4e37cf4b		VFMLA V23.S4, V26.S4, V11.S4
  kernel_arm64.go:150	0x100181034		4e37cf6a		VFMLA V23.S4, V27.S4, V10.S4
  kernel_arm64.go:151	0x100181038		4e37cf89		VFMLA V23.S4, V28.S4, V9.S4
  kernel_arm64.go:153	0x10018103c		4e38cf28		VFMLA V24.S4, V25.S4, V8.S4
  kernel_arm64.go:154	0x100181040		4e38cf47		VFMLA V24.S4, V26.S4, V7.S4
  kernel_arm64.go:155	0x100181044		4e38cf66		VFMLA V24.S4, V27.S4, V6.S4
  kernel_arm64.go:156	0x100181048		4e38cf80		VFMLA V24.S4, V28.S4, V0.S4
  kernel_arm64.go:48	0x10018104c		aa0703e5		MOVD R7, R5
  kernel_arm64.go:48	0x100181050		910010a7		ADD $4, R5, R7
  kernel_arm64.go:48	0x100181054		eb07003f		CMP R7, R1
  kernel_arm64.go:48	0x100181058		540006ab		BLT 53(PC)
  kernel_arm64.go:49	0x10018105c		eb07005f		CMP R7, R2
  kernel_arm64.go:49	0x100181060		54000d83		BCC 108(PC)
  kernel_arm64.go:49	0x100181064		eb0700bf		CMP R7, R5
  kernel_arm64.go:49	0x100181068		54000d28		BHI 105(PC)
  kernel_arm64.go:53	0x10018106c		d37ceca8		LSL $4, R5, R8
  kernel_arm64.go:53	0x100181070		9101010a		ADD $64, R8, R10
  kernel_arm64.go:49	0x100181074		8b05080b		ADD R5<<2, R0, R11
  kernel_arm64.go:50	0x100181078		8b05086d		ADD R5<<2, R3, R13
  kernel_arm64.go:51	0x10018107c		8b0508ce		ADD R5<<2, R6, R14
  kernel_arm64.go:52	0x100181080		8b05092f		ADD R5<<2, R9, R15
  kernel_arm64.go:53	0x100181084		eb01115f		CMP R1<<4, R10
  kernel_arm64.go:53	0x100181088		54000c08		BHI 96(PC)
  kernel_arm64.go:53	0x10018108c		eb05115f		CMP R5<<4, R10
  kernel_arm64.go:53	0x100181090		54fff142		BCS -118(PC)
  kernel_arm64.go:53	0x100181094		1400005c		JMP 92(PC)
  kernel_arm64.go:161	0x100181098		3dc00041		FMOVQ (R2), F1
  kernel_arm64.go:162	0x10018109c		3dc00442		FMOVQ 16(R2), F2
  kernel_arm64.go:163	0x1001810a0		3dc00843		FMOVQ 32(R2), F3
  kernel_arm64.go:164	0x1001810a4		3dc00c44		FMOVQ 48(R2), F4
  kernel_arm64.go:165	0x1001810a8		bc657815		FMOVS (R0)(R5<<2), F21
  kernel_arm64.go:170	0x1001810ac		bc657876		FMOVS (R3)(R5<<2), F22
  kernel_arm64.go:175	0x1001810b0		bc6578d7		FMOVS (R6)(R5<<2), F23
  kernel_arm64.go:180	0x1001810b4		bc657938		FMOVS (R9)(R5<<2), F24
  other_gen_arm64.go:67	0x1001810b8		4ea51cb9		VMOV V5.B16, V25.B16
  other_gen_arm64.go:67	0x1001810bc		6e0406b9		VMOV V21.S[0], V25.S[0]
  other_gen_arm64.go:67	0x1001810c0		4e040735		VDUP V25.S[0], V21.S4
  other_gen_arm64.go:67	0x1001810c4		4ea51cb9		VMOV V5.B16, V25.B16
  other_gen_arm64.go:67	0x1001810c8		6e0406d9		VMOV V22.S[0], V25.S[0]
  other_gen_arm64.go:67	0x1001810cc		4e040736		VDUP V25.S[0], V22.S4
  other_gen_arm64.go:67	0x1001810d0		4ea51cb9		VMOV V5.B16, V25.B16
  other_gen_arm64.go:67	0x1001810d4		6e0406f9		VMOV V23.S[0], V25.S[0]
  other_gen_arm64.go:67	0x1001810d8		4e040737		VDUP V25.S[0], V23.S4
  other_gen_arm64.go:67	0x1001810dc		4ea51cb9		VMOV V5.B16, V25.B16
  other_gen_arm64.go:67	0x1001810e0		6e040719		VMOV V24.S[0], V25.S[0]
  other_gen_arm64.go:67	0x1001810e4		4e040738		VDUP V25.S[0], V24.S4
  kernel_arm64.go:166	0x1001810e8		4e35cc34		VFMLA V21.S4, V1.S4, V20.S4
  kernel_arm64.go:167	0x1001810ec		4e35cc53		VFMLA V21.S4, V2.S4, V19.S4
  kernel_arm64.go:168	0x1001810f0		4e35cc72		VFMLA V21.S4, V3.S4, V18.S4
  kernel_arm64.go:169	0x1001810f4		4e35cc91		VFMLA V21.S4, V4.S4, V17.S4
  kernel_arm64.go:171	0x1001810f8		4e36cc30		VFMLA V22.S4, V1.S4, V16.S4
  kernel_arm64.go:172	0x1001810fc		4e36cc4f		VFMLA V22.S4, V2.S4, V15.S4
  kernel_arm64.go:173	0x100181100		4e36cc6e		VFMLA V22.S4, V3.S4, V14.S4
  kernel_arm64.go:174	0x100181104		4e36cc8d		VFMLA V22.S4, V4.S4, V13.S4
  kernel_arm64.go:176	0x100181108		4e37cc2c		VFMLA V23.S4, V1.S4, V12.S4
  kernel_arm64.go:177	0x10018110c		4e37cc4b		VFMLA V23.S4, V2.S4, V11.S4
  kernel_arm64.go:178	0x100181110		4e37cc6a		VFMLA V23.S4, V3.S4, V10.S4
  kernel_arm64.go:179	0x100181114		4e37cc89		VFMLA V23.S4, V4.S4, V9.S4
  kernel_arm64.go:181	0x100181118		4e38cc28		VFMLA V24.S4, V1.S4, V8.S4
  kernel_arm64.go:182	0x10018111c		4e38cc47		VFMLA V24.S4, V2.S4, V7.S4
  kernel_arm64.go:183	0x100181120		4e38cc66		VFMLA V24.S4, V3.S4, V6.S4
  kernel_arm64.go:184	0x100181124		4e38cc80		VFMLA V24.S4, V4.S4, V0.S4
  kernel_arm64.go:159	0x100181128		910004a5		ADD $1, R5, R5
  kernel_arm64.go:159	0x10018112c		eb05003f		CMP R5, R1
  kernel_arm64.go:159	0x100181130		5400016d		BLE 11(PC)
  kernel_arm64.go:160	0x100181134		d37ceca2		LSL $4, R5, R2
  kernel_arm64.go:160	0x100181138		91004047		ADD $16, R2, R7
  kernel_arm64.go:160	0x10018113c		eb0110ff		CMP R1<<4, R7
  kernel_arm64.go:160	0x100181140		54000608		BHI 48(PC)
  kernel_arm64.go:160	0x100181144		eb0510ff		CMP R5<<4, R7
  kernel_arm64.go:160	0x100181148		540005a3		BCC 45(PC)
  kernel_arm64.go:160	0x10018114c		8b051982		ADD R5<<6, R12, R2
  kernel_arm64.go:159	0x100181150		eb05003f		CMP R5, R1
  kernel_arm64.go:165	0x100181154		54fffa28		BHI -47(PC)
  kernel_arm64.go:165	0x100181158		14000028		JMP 40(PC)
  kernel_arm64.go:186	0x10018115c		f94033e0		MOVD 96(RSP), R0
  kernel_arm64.go:186	0x100181160		f1003c1f		CMP $15, R0
  kernel_arm64.go:186	0x100181164		54000489		BLS 36(PC)
  kernel_arm64.go:187	0x100181168		f9402fe0		MOVD 88(RSP), R0
  kernel_arm64.go:187	0x10018116c		3d800014		FMOVQ F20, (R0)
  kernel_arm64.go:188	0x100181170		3d800413		FMOVQ F19, 16(R0)
  kernel_arm64.go:189	0x100181174		3d800812		FMOVQ F18, 32(R0)
  kernel_arm64.go:190	0x100181178		3d800c11		FMOVQ F17, 48(R0)
  kernel_arm64.go:191	0x10018117c		f9403fe0		MOVD 120(RSP), R0
  kernel_arm64.go:191	0x100181180		f1003c1f		CMP $15, R0
  kernel_arm64.go:191	0x100181184		54000369		BLS 27(PC)
  kernel_arm64.go:192	0x100181188		f9403be0		MOVD 112(RSP), R0
  kernel_arm64.go:192	0x10018118c		3d800010		FMOVQ F16, (R0)
  kernel_arm64.go:193	0x100181190		3d80040f		FMOVQ F15, 16(R0)
  kernel_arm64.go:194	0x100181194		3d80080e		FMOVQ F14, 32(R0)
  kernel_arm64.go:195	0x100181198		3d800c0d		FMOVQ F13, 48(R0)
  kernel_arm64.go:196	0x10018119c		f9404be0		MOVD 144(RSP), R0
  kernel_arm64.go:196	0x1001811a0		f1003c1f		CMP $15, R0
  kernel_arm64.go:196	0x1001811a4		54000249		BLS 18(PC)
  kernel_arm64.go:197	0x1001811a8		f94047e0		MOVD 136(RSP), R0
  kernel_arm64.go:197	0x1001811ac		3d80000c		FMOVQ F12, (R0)
  kernel_arm64.go:198	0x1001811b0		3d80040b		FMOVQ F11, 16(R0)
  kernel_arm64.go:199	0x1001811b4		3d80080a		FMOVQ F10, 32(R0)
  kernel_arm64.go:200	0x1001811b8		3d800c09		FMOVQ F9, 48(R0)
  kernel_arm64.go:201	0x1001811bc		f94057e0		MOVD 168(RSP), R0
  kernel_arm64.go:201	0x1001811c0		f1003c1f		CMP $15, R0
  kernel_arm64.go:201	0x1001811c4		54000129		BLS 9(PC)
  kernel_arm64.go:202	0x1001811c8		f94053e0		MOVD 160(RSP), R0
  kernel_arm64.go:202	0x1001811cc		3d800008		FMOVQ F8, (R0)
  kernel_arm64.go:203	0x1001811d0		3d800407		FMOVQ F7, 16(R0)
  kernel_arm64.go:204	0x1001811d4		3d800806		FMOVQ F6, 32(R0)
  kernel_arm64.go:205	0x1001811d8		3d800c00		FMOVQ F0, 48(R0)
  kernel_arm64.go:206	0x1001811dc		f85f83fd		MOVD -8(RSP), R29
  kernel_arm64.go:206	0x1001811e0		f84507fe		MOVD.P 80(RSP), R30
  kernel_arm64.go:206	0x1001811e4		d65f03c0		RET
  kernel_arm64.go:201	0x1001811e8		97fc11ea		CALL runtime.panicBounds(SB)
  kernel_arm64.go:196	0x1001811ec		97fc11e9		CALL runtime.panicBounds(SB)
  kernel_arm64.go:191	0x1001811f0		97fc11e8		CALL runtime.panicBounds(SB)
  kernel_arm64.go:186	0x1001811f4		97fc11e7		CALL runtime.panicBounds(SB)
  kernel_arm64.go:165	0x1001811f8		97fc11e6		CALL runtime.panicBounds(SB)
  kernel_arm64.go:160	0x1001811fc		97fc11e5		CALL runtime.panicBounds(SB)
  kernel_arm64.go:160	0x100181200		97fc11e4		CALL runtime.panicBounds(SB)
  kernel_arm64.go:53	0x100181204		97fc11e3		CALL runtime.panicBounds(SB)
  kernel_arm64.go:53	0x100181208		97fc11e2		CALL runtime.panicBounds(SB)
  kernel_arm64.go:49	0x10018120c		97fc11e1		CALL runtime.panicBounds(SB)
  kernel_arm64.go:49	0x100181210		97fc11e0		CALL runtime.panicBounds(SB)
  kernel_arm64.go:38	0x100181214		97fc11df		CALL runtime.panicBounds(SB)
  kernel_arm64.go:37	0x100181218		97fc11de		CALL runtime.panicBounds(SB)
  kernel_arm64.go:37	0x10018121c		97fc11dd		CALL runtime.panicBounds(SB)
  kernel_arm64.go:37	0x100181220		97fc11dc		CALL runtime.panicBounds(SB)
  kernel_arm64.go:37	0x100181224		d503201f		NOOP
  kernel_arm64.go:35	0x100181228		a90687e0		STP (R0, R1), 104(RSP)
  kernel_arm64.go:35	0x10018122c		a9078fe2		STP (R2, R3), 120(RSP)
  kernel_arm64.go:35	0x100181230		a90897e4		STP (R4, R5), 136(RSP)
  kernel_arm64.go:35	0x100181234		a9099fe6		STP (R6, R7), 152(RSP)
  kernel_arm64.go:35	0x100181238		a90aa7e8		STP (R8, R9), 168(RSP)
  kernel_arm64.go:35	0x10018123c		a90bafea		STP (R10, R11), 184(RSP)
  kernel_arm64.go:35	0x100181240		a90cb7ec		STP (R12, R13), 200(RSP)
  kernel_arm64.go:35	0x100181244		f9006fee		MOVD R14, 216(RSP)
  kernel_arm64.go:35	0x100181248		aa1e03e3		MOVD R30, R3
  kernel_arm64.go:35	0x10018124c		97fc098d		CALL runtime.morestack_noctxt.abi0(SB)
  kernel_arm64.go:35	0x100181250		a94687e0		LDP 104(RSP), (R0, R1)
  kernel_arm64.go:35	0x100181254		a9478fe2		LDP 120(RSP), (R2, R3)
  kernel_arm64.go:35	0x100181258		a94897e4		LDP 136(RSP), (R4, R5)
  kernel_arm64.go:35	0x10018125c		a9499fe6		LDP 152(RSP), (R6, R7)
  kernel_arm64.go:35	0x100181260		a94aa7e8		LDP 168(RSP), (R8, R9)
  kernel_arm64.go:35	0x100181264		a94bafea		LDP 184(RSP), (R10, R11)
  kernel_arm64.go:35	0x100181268		a94cb7ec		LDP 200(RSP), (R12, R13)
  kernel_arm64.go:35	0x10018126c		f9406fee		MOVD 216(RSP), R14
  kernel_arm64.go:35	0x100181270		17fffecc		JMP github.com/GetStream/gophonic/internal/whispergemm.kernel4x16(SB)
  kernel_arm64.go:35	0x100181274		00000000		?
  kernel_arm64.go:35	0x100181278		00000000		?
  kernel_arm64.go:35	0x10018127c		00000000		?
