//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

!s = !clift.struct<
  "/type-definition/1-StructDefinition" : size(1) {
    offset(0) : !clift.union<
      "/type-definition/2-UnionDefinition" : {
        !clift.struct<"/type-definition/1-StructDefinition">
      }
    >
  }
>

// CHECK: recursive class type
module attributes {clift.module, clift.test = !s} {}
