//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s --terminal-branch-complement-hoisting | FileCheck %s

!void = !clift.void
!int32_t = !clift.int<signed 4>

!f = !clift.func<"" : !void(!int32_t)>

// No branch falls through, so precedence hoists the default for its continue.
// The break following the switch is then unreachable, and the NoFallthrough
// trait does not allow it after the hoisted continue either, so it is erased.

module attributes {clift.module} {
  // CHECK-LABEL: clift.func @f
  clift.func @f<!f>(%arg0 : !int32_t) -> !void {
    %break = clift.make_label
    %continue = clift.make_label
    clift.while break %break continue %continue cond {
      %c = clift.imm 1 : !int32_t
      clift.yield %c : !int32_t
    } body {
      // CHECK: clift.switch {
      clift.switch {
        clift.yield %arg0 : !int32_t
      // CHECK: } case 0 {
      // CHECK-NEXT: clift.return
      } case 0 {
        clift.return {}
      // CHECK-NOT: default
      } default {
        clift.continue_to %continue
      }
      // CHECK: clift.continue_to
      // The break is erased, so nothing follows the hoisted continue.
      // CHECK-NOT: clift.break_to
      clift.break_to %break
    }
  }
}
