//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s --promote-break-continue | FileCheck %s

!void = !clift.void

!f = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" : !void()
>

// A break_to/continue_to targeting an *outer* loop cannot be promoted: a plain
// break/continue would bind to the inner (innermost) loop instead, so the
// operand must be kept.

module attributes {clift.module} {
  clift.func @f<!f>() {
    %break = clift.make_label
    %continue = clift.make_label

    clift.for break %break continue %continue body {
      clift.for body {
        // CHECK: clift.break_to %
        clift.break_to %break
      }

      clift.for body {
        // CHECK: clift.continue_to %
        clift.continue_to %continue
      }
    }
  }
}
