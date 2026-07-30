//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s --promote-break-continue | FileCheck %s

!void = !clift.void
!int32_t = !clift.int<signed 4>

!f = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" : !void()
>

// A switch is transparent to continue but captures break. Hence a continue_to
// nested within a switch can still be promoted, while a break_to targeting the
// enclosing loop cannot (a plain break would break the switch instead).
//
// The continue label thus loses its only jump user and is dropped from the loop,
// while the break label is kept because its break_to is still labeled.

module attributes {clift.module} {
  clift.func @f<!f>() {
    %break = clift.make_label
    %continue = clift.make_label

    // The loop keeps its break label but loses the (now dead) continue label.
    // CHECK: clift.for break %{{[a-zA-Z0-9_]+}} body {
    clift.for break %break continue %continue body {
      clift.switch {
        %0 = clift.imm 0 : !int32_t
        clift.yield %0 : !int32_t
      } case 0 {
        // CHECK: clift.continue_to{{$}}
        clift.continue_to %continue
      } default {
        // CHECK: clift.break_to %
        clift.break_to %break
      }
    }
  }
}
