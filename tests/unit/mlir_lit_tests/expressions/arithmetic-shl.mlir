//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s | FileCheck %s

!int16_t = !clift.primitive<SignedKind 2>
!int16_t$const = !clift.primitive<is_const = true, SignedKind 2>

!int32_t = !clift.primitive<SignedKind 4>
!int32_t$const = !clift.primitive<is_const = true, SignedKind 4>

// CHECK: [[M16:%[0-9]+]] = clift.undef : !int16_t
%m16 = clift.undef : !int16_t
// CHECK: [[M32:%[0-9]+]] = clift.undef : !int32_t
%m32 = clift.undef : !int32_t

// CHECK: [[C16:%[0-9]+]] = clift.undef : !int16_t$const
%c16 = clift.undef : !int16_t$const
// CHECK: [[C32:%[0-9]+]] = clift.undef : !int32_t$const
%c32 = clift.undef : !int32_t$const

// CHECK: clift.shl [[M32]], [[M32]]
clift.shl %m32, %m32 : !int32_t
// CHECK: clift.shl [[M32]], [[M32]]
clift.shl %m32, %m32 : (!int32_t, !int32_t)
// CHECK: clift.shl [[M32]], [[M16]]
clift.shl %m32, %m16 : (!int32_t, !int16_t)

// CHECK: clift.shl [[M32]], [[C32]]
clift.shl %m32, %c32 : (!int32_t, !int32_t$const)
// CHECK: clift.shl [[M32]], [[C16]]
clift.shl %m32, %c16 : (!int32_t, !int16_t$const)
// CHECK: clift.shl [[C32]], [[M32]]
clift.shl %c32, %m32 : (!int32_t$const, !int32_t)
// CHECK: clift.shl [[C16]], [[M32]]
clift.shl %c16, %m32 : (!int16_t$const, !int32_t)
