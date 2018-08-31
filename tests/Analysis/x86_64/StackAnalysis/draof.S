#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

.intel_syntax noprefix
.global _start
_start:
    call draof
    ret

# rax is NoOrDead
draof:
    mov rax,42
    ret
