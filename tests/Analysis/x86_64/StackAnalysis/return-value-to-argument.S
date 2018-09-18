#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

.intel_syntax noprefix
.global _start
_start:
    call return_value_to_argument
    ret

return_value_to_argument:
    call return_rax
    call rax_not_my_argument
    ret

return_rax:
    mov rax, 42
    ret

rax_not_my_argument:
    mov rax, 43
    ret
