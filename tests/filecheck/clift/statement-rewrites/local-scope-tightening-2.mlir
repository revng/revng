//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s --tighten-variable-scopes | FileCheck %s

!void = !clift.void
!int32_t = !clift.int<signed 4>

!f = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" : !void()
>

// Local is not moved due to two users in directly nested regions.

module attributes {clift.module} {
  clift.func @f<!f>() {
    // CHECK: [[L:%[0-9]+]] = clift.local : !int32_t
    %L = clift.local : !int32_t
    // CHECK: clift.if {
    clift.if {
      %0 = clift.true
      clift.yield %0 : !clift.bool
    // CHECK: } then {
    } then {
      // CHECK: clift.expr {
      clift.expr {
        // CHECK: clift.yield [[L]] : !int32_t
        clift.yield %L : !int32_t
      // CHECK: }
      }
    // CHECK: } else {
    } else {
      // CHECK: clift.expr {
      clift.expr {
        // CHECK: clift.yield [[L]] : !int32_t
        clift.yield %L : !int32_t
      // CHECK: }
      }
    }
  // CHECK: }
  }
}
