#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

.intel_syntax noprefix
.global _start
_start:
    call saofc
    ret

saofc:
    call callee
    # rax is a No here, since the caller has to assume that the callee will
    # clobber all the stack arguments. Therefore if the caller reads it, it must
    # mean that is *not* an argument
    mov rax,[rsp+8]
    ret

callee:
    ret
