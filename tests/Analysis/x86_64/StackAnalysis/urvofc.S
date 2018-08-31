#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

.intel_syntax noprefix
.global _start
_start:
    call urvofc
    ret

# rax is a return value
urvofc:
    call callee
    mov rbx,rax
    ret

callee:
    mov rax, 42
    mov rbx, rax
    ret
