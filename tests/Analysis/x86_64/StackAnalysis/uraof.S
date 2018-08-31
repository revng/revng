#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

.intel_syntax noprefix
.global _start
_start:
    call uraof
    ret

# rax is an argument.
uraof:
    mov rbx,rax
    ret
