//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s --promote-break-continue | FileCheck %s

!void = !clift.void

!f = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" : !void()
>

// A break_to/continue_to targeting the break/continue label of its immediately
// enclosing loop is promoted to an operand-less break/continue. Once the label
// has no jump users left it is dead, so it is dropped from the loop and its
// make_label is erased.

module attributes {clift.module} {
  clift.func @f<!f>() {
    %break = clift.make_label
    %continue = clift.make_label

    // CHECK-NOT: clift.make_label

    // CHECK: clift.for body {
    clift.for break %break body {
      // CHECK-NEXT: clift.break_to{{$}}
      clift.break_to %break
    }

    // CHECK: clift.for body {
    clift.for continue %continue body {
      // CHECK-NEXT: clift.continue_to{{$}}
      clift.continue_to %continue
    }
  }
}
