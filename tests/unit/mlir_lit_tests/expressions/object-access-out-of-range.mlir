//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

!int32_t = !clift.primitive<signed 4>

!s = !clift.struct<
  "/type-definition/1-StructDefinition" : size(4) {
    "/type-definition/1-StructDefinition/0" : offset(0) !int32_t
  }
>

%0 = clift.undef : !s

// CHECK: struct or union member index out of range
%1 = clift.access<1> %0 : !s -> !int32_t
