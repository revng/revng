//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s --tighten-variable-scopes | FileCheck %s

!void = !clift.void
!int32_t = !clift.int<signed 4>

!f = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" : !void()
>

module attributes {clift.module} {
  clift.func @f<!f>() {
    // CHECK: clift.make_label
    %G = clift.make_label

    // CHECK-NOT: clift.local
    %L = clift.local : !int32_t

    // CHECK: clift.if {
    clift.if {
      %0 = clift.imm 1 : !int32_t
      clift.yield %0 : !int32_t
    // CHECK: } then {
    } then {
      // clift.assign_label
      clift.assign_label %G
    // CHECK: }
    }

    // CHECK: [[L:%[0-9]+]] = clift.local : !int32_t
    // CHECK: clift.expr {
    clift.expr {
      // CHECK: clift.yield [[L]] : !int32_t
      clift.yield %L : !int32_t
    // CHECK: }
    }

    // CHECK: clift.if {
    clift.if {
      %0 = clift.imm 2 : !int32_t
      clift.yield %0 : !int32_t
    // CHECK: } then {
    } then {
      // clift.goto
      clift.goto %G
    // CHECK: }
    }
  // CHECK: }
  }
}
