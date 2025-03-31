//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

!int32_t = !clift.primitive<signed 4>
!uint32_t = !clift.primitive<unsigned 4>

!s = !clift.struct<
  "/type-definition/1-StructDefinition" : size(4) {
    offset(0) as "x" : !int32_t
  }
>

%0 = clift.undef : !s

// CHECK: result type must match the selected member type
%1 = clift.access<0> %0 : !s -> !uint32_t
