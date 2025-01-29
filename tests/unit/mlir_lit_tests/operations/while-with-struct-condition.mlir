//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

!void = !clift.primitive<VoidKind 0>

!s = !clift.defined<#clift.struct<unique_handle = "/model-type/1", name = "", size = 1, fields = []>>

// CHECK: condition requires a scalar type
clift.while {
  %0 = clift.undef : !s
  clift.yield %0 : !s
} {}
