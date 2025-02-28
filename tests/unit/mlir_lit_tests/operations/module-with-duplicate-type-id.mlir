//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

!b = !clift.primitive<unsigned 1>

!s = !clift.defined<#clift.struct<"/model-type/1" : size(1) {}>>
!u = !clift.defined<#clift.union<"/model-type/1" : { !b }>>

// CHECK: two distinct type definitions with the same unique handle: '/model-type/1'
clift.module {
} {
  s = !s,
  u = !u
}
