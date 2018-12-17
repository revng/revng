#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

    .intel_syntax noprefix
	.globl	_start
_start:
    cmp    DWORD PTR [rbx+0x8],0x3
    ja     end
    mov    eax,DWORD PTR [rbx+0x8]
    jmp    QWORD PTR [rax*8+jumptable]
one:
    ret
two:
    ret
three:
    ret
end:
    ret

    .section        .rodata
jumptable:
    .quad one
    .quad two
    .quad three
    .quad end
