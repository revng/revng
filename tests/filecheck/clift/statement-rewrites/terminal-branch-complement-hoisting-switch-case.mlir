//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s --terminal-branch-complement-hoisting | FileCheck %s

!void = !clift.void
!int32_t = !clift.int<signed 4>

!f = !clift.func<"" : !void(!int32_t)>

// The only fall-through branch is a case (the others and the default are
// non-fallthrough), so its body is hoisted after the switch. The case is kept
// as an empty body so its label still falls through to the hoisted code.

module attributes {clift.module} {
  // CHECK-LABEL: clift.func @f
  clift.func @f<!f>(%arg0 : !int32_t) -> !void {
    // CHECK: clift.switch {
    clift.switch {
      clift.yield %arg0 : !int32_t
    // CHECK: } case 0 {
    // CHECK-NEXT: } case 1 {
    } case 0 {
      clift.expr {
        %0 = clift.imm 20 : !int32_t
        clift.yield %0 : !int32_t
      }
    } case 1 {
      // CHECK: clift.return
      clift.return {}
    // CHECK: } default {
    } default {
      // CHECK: clift.return
      clift.return {}
    }
    // CHECK: clift.expr {
    // CHECK: clift.imm 20 : !int32_t
  }
}
