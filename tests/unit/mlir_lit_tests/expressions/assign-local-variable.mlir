//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s

!int32_t = !clift.primitive<signed 4>
!int32_t$const = !clift.const<!clift.primitive<signed 4>>

%rvalue = clift.undef : !int32_t
%rvalue_const = clift.undef : !int32_t$const

%lvalue = clift.local !int32_t
%assign = clift.assign %lvalue, %rvalue : !int32_t
clift.assign %assign, %lvalue : !int32_t
clift.assign %assign, %rvalue_const : (!int32_t, !int32_t$const)
