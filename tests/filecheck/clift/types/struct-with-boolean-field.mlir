//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %root/bin/revng clift-opt %s 2>&1 | FileCheck %s

!void = !clift.void

// CHECK: field types must be object types
!s = !clift.struct<
  "/type-definition/1-StructDefinition" : size(1) {
    "/struct-field/1-StructDefinition/0" : offset(0) !clift.bool
  }
>
