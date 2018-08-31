#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

.intel_syntax noprefix
.global _start
_start:
    call func
    ret

# In this function call rax is an argument, but in the callee is NoOrDead,
# therefore, rax is Dead in callee.
func:
    mov rax,42
    call callee
    ret

callee:
    mov rax,43
    ret
