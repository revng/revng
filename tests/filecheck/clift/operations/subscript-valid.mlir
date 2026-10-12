//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s | FileCheck %s

// The result of a subscript has to be exactly the pointee type of the pointer
// operand, and the index has to be an integer. These are the shapes that
// satisfy both.

!int32_t = !clift.int<signed 4>
!uint64_t = !clift.int<unsigned 8>
!const_int32_t = !clift.const<!clift.int<signed 4>>
!p = !clift.ptr<8 to !int32_t>
!p_const = !clift.ptr<8 to !const_int32_t>
!pp = !clift.ptr<8 to !p>
!p_alias = !clift.typedef<
  "/type-definition/2-TypedefDefinition" as "p_alias" : !p
>
!int_alias = !clift.typedef<
  "/type-definition/3-TypedefDefinition" as "int_alias" : !int32_t
>
!p_to_alias = !clift.ptr<8 to !int_alias>

%i = clift.undef : !int32_t
%u = clift.undef : !uint64_t

// A signed index, and an unsigned one of a different width.
%0 = clift.undef : !p
// CHECK: clift.subscript %{{.*}} : (!clift.ptr<8 to !int32_t>, !int32_t)
%1 = clift.subscript %0, %i : (!p, !int32_t)
// CHECK: clift.subscript %{{.*}} : (!clift.ptr<8 to !int32_t>, !uint64_t)
%2 = clift.subscript %0, %u : (!p, !uint64_t)

// A pointer to a const scalar: the result keeps the qualifier.
%3 = clift.undef : !p_const
// CHECK: clift.subscript %{{.*}} : (!clift.ptr<8 to !clift.const<!int32_t>>, !int32_t)
%4 = clift.subscript %3, %i : (!p_const, !int32_t)

// A pointer to a pointer: the result is the inner pointer.
%5 = clift.undef : !pp
// CHECK: clift.subscript %{{.*}} : (!clift.ptr<8 to !clift.ptr<8 to !int32_t>>, !int32_t)
%6 = clift.subscript %5, %i : (!pp, !int32_t)

// The pointer operand is a typedef of a pointer, so reaching the pointee
// unwraps it.
%7 = clift.undef : !p_alias
// CHECK: clift.subscript %{{.*}} : (!p_alias, !int32_t)
%8 = clift.subscript %7, %i : (!p_alias, !int32_t)

// The pointee is itself a typedef: the result is that typedef, not the type it
// stands for.
%9 = clift.undef : !p_to_alias
// CHECK: clift.subscript %{{.*}} : (!clift.ptr<8 to !int_alias>, !int32_t)
%10 = clift.subscript %9, %i : (!p_to_alias, !int32_t)
