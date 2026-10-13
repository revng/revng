//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s | FileCheck %s

// The result of an indirection has to be exactly the pointee type of its
// operand. These are the shapes that satisfy it.

!int32_t = !clift.int<signed 4>
!const_int32_t = !clift.const<!clift.int<signed 4>>
!p = !clift.ptr<8 to !int32_t>
!pp = !clift.ptr<8 to !p>
!p_const = !clift.ptr<8 to !const_int32_t>
!p_alias = !clift.typedef<
  "/type-definition/2-TypedefDefinition" as "p_alias" : !p
>
!int_alias = !clift.typedef<
  "/type-definition/3-TypedefDefinition" as "int_alias" : !int32_t
>
!p_to_alias = !clift.ptr<8 to !int_alias>

// A pointer to a scalar.
%0 = clift.undef : !p
// CHECK: clift.indirection %{{.*}} : !clift.ptr<8 to !int32_t>
%1 = clift.indirection %0 : !p

// A pointer to a pointer: the result is the inner pointer, not its pointee.
%2 = clift.undef : !pp
// CHECK: clift.indirection %{{.*}} : !clift.ptr<8 to !clift.ptr<8 to !int32_t>>
%3 = clift.indirection %2 : !pp

// A pointer to a const scalar: the result keeps the qualifier.
%4 = clift.undef : !p_const
// CHECK: clift.indirection %{{.*}} : !clift.ptr<8 to !clift.const<!int32_t>>
%5 = clift.indirection %4 : !p_const

// The operand is a typedef of a pointer, so reaching the pointee unwraps it.
%6 = clift.undef : !p_alias
// CHECK: clift.indirection %{{.*}} : !p_alias
%7 = clift.indirection %6 : !p_alias

// The pointee is itself a typedef: the result is that typedef, not the type it
// stands for.
%8 = clift.undef : !p_to_alias
// CHECK: clift.indirection %{{.*}} : !clift.ptr<8 to !int_alias>
%9 = clift.indirection %8 : !p_to_alias
