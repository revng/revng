//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

!s = !clift.struct<
  "/type-definition/1-StructDefinition" : size(1) {}
>

!u = !clift.union<
  "/type-definition/1-StructDefinition" : {
    "/struct-field/1-StructDefinition/0" : !s
  }
>

// CHECK: two distinct type definitions with the same handle: '/type-definition/1-StructDefinition'
module attributes {clift.module, clift.test = !u} {}
