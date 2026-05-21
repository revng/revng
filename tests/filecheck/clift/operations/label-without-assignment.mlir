//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %root/bin/revng clift-opt %s 2>&1 | FileCheck %s

!void = !clift.void

!f = !clift.func<
  "/type-definition/1-CABIFunctionDefinition" : !void()
>

clift.func @f<!f>() {
  // CHECK: clift.make_label with a use by a jump operation must have an assignment
  %label = clift.make_label
  clift.goto %label
}
