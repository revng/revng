//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s

!void = !clift.void

!f = !clift.func<
  "/type-definition/1-CABIFunctionDefinition" : !void()
>

clift.func @f<!f>() {
  %label = clift.make_label
  clift.goto %label
  clift.assign_label %label
}
