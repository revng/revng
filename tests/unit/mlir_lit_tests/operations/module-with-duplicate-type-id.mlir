//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

!b = !clift.primitive<unsigned 1>

!s = !clift.struct<
  "/type-definition/1-StructDefinition" : size(1) {}
>

!u = !clift.union<
  "/type-definition/1-StructDefinition" : { !b }
>

// CHECK: two distinct type definitions with the same unique handle: '/type-definition/1-StructDefinition'
clift.module {
} {
  s = !s,
  u = !u
}
