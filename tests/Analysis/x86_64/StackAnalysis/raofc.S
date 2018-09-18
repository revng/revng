#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

.intel_syntax noprefix
.global _start
_start:
    call raofc
    ret

# We cannot say rax is an argument, since it's unused in the callee
raofc:
    mov rax,42
    call callee
    ret

callee:
    ret
