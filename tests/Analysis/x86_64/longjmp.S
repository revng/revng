#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

.intel_syntax noprefix
.global _start
_start:
    call longjmp
    ret

longjmp:
    jle ret
    push rax
    jmp rax
ret:
    ret
