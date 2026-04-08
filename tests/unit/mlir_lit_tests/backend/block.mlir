//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt --emit-c %s | FileCheck %s
// RUN: %revngcliftopt --emit-c=ptml %s -o /dev/null | %revngptml | FileCheck %s

!void = !clift.void
!int32_t = !clift.int<signed 4>

!f = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" : !void()
>

module attributes {clift.module} {
  // CHECK: void fun_0x40001001(void) {
  clift.func @fun_0x40001001<!f>() attributes {
    handle = "/function/0x40001001:Code_x86_64"
  } {
    // CHECK: {
    clift.block {
      // CHECK: 1;
      clift.expr {
        %0 = clift.imm 1 : !int32_t
        clift.yield %0 : !int32_t
      }
    // CHECK: }
    }

    // CHECK: if (2) {
    clift.if {
      %0 = clift.imm 2 : !int32_t
      clift.yield %0 : !int32_t
    } then {
      clift.block {
        // CHECK: {
        // CHECK-NOT: {
        clift.block {
          // CHECK: 3;
          clift.expr {
            %0 = clift.imm 3 : !int32_t
            clift.yield %0 : !int32_t
          }
        // CHECK: }
        }
      }
    // CHECK: } else {
    } else {
      // CHECK: 4;
      clift.expr {
        %0 = clift.imm 4 : !int32_t
        clift.yield %0 : !int32_t
      }
    // CHECK: }
    }
  }
  // CHECK: }
}
