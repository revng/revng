//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s

!int32_t = !clift.primitive<SignedKind 4>
!int32_t$ptr = !clift.pointer<pointee_type = !int32_t, pointer_size = 8>

%ptr = clift.undef : !int32_t$ptr
%rvalue = clift.imm 0 : !int32_t
%lvalue = clift.subscript %ptr, %rvalue : (!int32_t$ptr, !int32_t)
clift.addressof %lvalue : !int32_t$ptr
