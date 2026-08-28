//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s --terminal-branch-complement-hoisting | FileCheck %s

!void = !clift.void
!int32_t = !clift.int<signed 4>

!f = !clift.func<"" : !void(!int32_t)>

// Neither branch falls through, so the else is hoisted and its goto becomes
// part of the nesting scope. That makes the statement following the if
// unreachable, and the NoFallthrough trait does not allow it to stay after the
// hoisted goto either, so it is erased.
//
// The CHECK-NEXT chain pins the whole tail of the function: the hoisted goto is
// the last statement, with neither an else nor the erased goto left behind.

// CHECK: module attributes {clift.module} {
module attributes {clift.module} {
  // CHECK: clift.func
  // CHECK-SAME: {
  clift.func @f<!f>(%arg0 : !int32_t) -> !void {
    %0 = clift.make_label
    clift.assign_label %0
    // CHECK: clift.if {
    clift.if {
      // CHECK: clift.yield %arg0 : !int32_t
      clift.yield %arg0 : !int32_t
    // CHECK: } then {
    } then {
      // CHECK-NEXT: clift.goto %0
      clift.goto %0
    // CHECK-NEXT: }
    } else {
      clift.goto %0
    }
    // CHECK-NEXT: clift.goto %0
    // CHECK-NEXT: }
    // CHECK-NEXT: }
    clift.goto %0
  }
}
