#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

.intel_syntax noprefix
.global _start
_start:
    call urvof
    ret

urvof:
    # This is a dead assingment to a register, therefore it has to be a return
    # value
    mov rax,42
    ret
