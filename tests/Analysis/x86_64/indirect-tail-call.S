#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

.intel_syntax noprefix
.global _start
_start:
    call tailcall
    ret

tailcall:
    push rax
    pop rax
    jmp rax
