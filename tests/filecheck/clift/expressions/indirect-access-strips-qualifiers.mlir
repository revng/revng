//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %root/bin/revng clift-opt %s 2>&1 | FileCheck %s

!int32_t = !clift.int<signed 4>

!s = !clift.struct<
  "/type-definition/1-StructDefinition" as "s" : size(4) {
    "/struct-field/1-StructDefinition/0" as "x" : offset(0) !int32_t
  }
>

%s = clift.undef : !clift.ptr<8 to !clift.const<!s>>

// CHECK: result type must match the accessed member type
clift.ptr_access<0> %s : !clift.ptr<8 to !clift.const<!s>> -> !int32_t
