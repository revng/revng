//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s | FileCheck %s

!int32_t = !clift.primitive<signed 4>
!int32_t$const = !clift.primitive<const signed 4>

// CHECK: [[M:%[0-9]+]] = clift.undef : !int32_t
%m = clift.undef : !int32_t

// CHECK: [[C:%[0-9]+]] = clift.undef : !int32_t$const
%c = clift.undef : !int32_t$const

// CHECK: clift.neg [[M]] : !int32_t{{$}}
clift.neg %m : !int32_t

// CHECK: clift.neg [[C]] : !int32_t$const{{$}}
clift.neg %c : !int32_t$const
