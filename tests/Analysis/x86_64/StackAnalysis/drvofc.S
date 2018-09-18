#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

.intel_syntax noprefix
.global _start
_start:
    call drvofc
    ret

# In this function call, rax is NoOrDead
drvofc:
    call rbx
    mov rax,42
    ret
