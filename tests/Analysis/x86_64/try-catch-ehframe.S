#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

#
# Generated from the following C++ code:
#
# extern "C" void function_call();
#
# int main() {
#   try {
#     function_call();
#   } catch(int) {
#     return 1;
#   }
#
#   return 0;
# }
#

	.intel_syntax noprefix
	.text

    # Placeholders for not included functions
__gxx_personality_v0:
_Unwind_Resume:
__cxa_begin_catch:
__cxa_end_catch:
_ZTIi:
function_call:
    ret

	.globl	_start
	.type	_start, @function
_start:
.LFB0:
	.cfi_startproc
	.cfi_personality 0x3,__gxx_personality_v0
	.cfi_lsda 0x3,.LLSDA0
	push	rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	mov	rbp, rsp
	.cfi_def_cfa_register 6
	push	rbx
	sub	rsp, 24
	.cfi_offset 3, -24
.LEHB0:
	call	function_call
.LEHE0:
	mov	ebx, 0
.L2:
	mov	eax, ebx
	jmp	.L7
.L6:
	cmp	rdx, 1
	je	.L5
	mov	rdi, rax
.LEHB1:
	call	_Unwind_Resume
.LEHE1:
.L5:
	mov	rdi, rax
	call	__cxa_begin_catch
	mov	eax, DWORD PTR [rax]
	mov	DWORD PTR [rbp-20], eax
	mov	ebx, 1
	call	__cxa_end_catch
	jmp	.L2
.L7:
	add	rsp, 24
	pop	rbx
	pop	rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.globl	__gxx_personality_v0
	.section	.gcc_except_table,"a",@progbits
	.align 4
.LLSDA0:
	.byte	0xff
	.byte	0x3
	.uleb128 .LLSDATT0-.LLSDATTD0
.LLSDATTD0:
	.byte	0x1
	.uleb128 .LLSDACSE0-.LLSDACSB0
.LLSDACSB0:
	.uleb128 .LEHB0-.LFB0
	.uleb128 .LEHE0-.LEHB0
	.uleb128 .L6-.LFB0
	.uleb128 0x1
	.uleb128 .LEHB1-.LFB0
	.uleb128 .LEHE1-.LEHB1
	.uleb128 0
	.uleb128 0
.LLSDACSE0:
	.byte	0x1
	.byte	0
	.align 4
	.long	_ZTIi
.LLSDATT0:
	.text
	.size	_start, .-_start
