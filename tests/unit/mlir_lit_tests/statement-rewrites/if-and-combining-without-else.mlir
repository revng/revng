//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s --optimize-statements="enable-patterns=if-and-combining" | FileCheck %s

!void = !clift.void
!int32_t = !clift.int<signed 4>
!f = !clift.func<"/model-type/1001" : !void()>

// CHECK: module attributes {clift.module} {
module attributes {clift.module} {
  // CHECK: clift.func
  // CHECK-SAME: {
  clift.func @f<!f>() -> !void {
    // CHECK: %0 = clift.make_label
    %0 = clift.make_label
    // CHECK: clift.if {
    clift.if {
      // CHECK: %1 = clift.imm 1 : !int32_t
      %1 = clift.imm 1 : !int32_t
      // CHECK: %2 = clift.imm 2 : !int32_t
      // CHECK: %3 = clift.and %1, %2 : !int32_t -> !int8_t
      // CHECK: clift.yield %3 : !int8_t
      clift.yield %1 : !int32_t
    // CHECK: } then {
    } then {
      // CHECK-NOT: clift.if
      clift.if {
        %1 = clift.imm 2 : !int32_t
        clift.yield %1 : !int32_t
      } then {
        // CHECK-NEXT: clift.expr {
        clift.expr {
          // CHECK: %1 = clift.imm 3 : !int32_t
          %1 = clift.imm 3 : !int32_t
          // CHECK: clift.yield %1 : !int32_t
          clift.yield %1 : !int32_t
        // CHECK: }
        }
      }
    // CHECK-NEXT: }
    }
    // CHECK-NOT: else
  // CHECK: }
  }
// CHECK: }
}
