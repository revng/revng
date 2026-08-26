//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %root/bin/revng clift-opt %s 2>&1 | FileCheck %s

!void = !clift.void

!s = !clift.struct<
  "/type-definition/1-StructDefinition" : size(1) {}
>

// CHECK: failed to verify constraint: region representing a condition expression
clift.while cond {
  %0 = clift.undef : !s
  clift.yield %0 : !s
} body {
}
