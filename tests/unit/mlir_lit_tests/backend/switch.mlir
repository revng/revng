//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt --emit-c %s -o /dev/null | FileCheck %s
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
    // CHECK-NEXT: switch (0) {
    clift.switch {
      %0 = clift.imm 0 : !int32_t
      clift.yield %0 : !int32_t
    // CHECK-NEXT: case 0:
    } case 0 {
      // CHECK-NEXT: 1;
      clift.expr {
        %1 = clift.imm 1 : !int32_t
        clift.yield %1 : !int32_t
      }
    // CHECK-NEXT: break;
    // CHECK-NEXT: default:
    } default {
      // CHECK-NEXT: 2;
      clift.expr {
        %2 = clift.imm 2 : !int32_t
        clift.yield %2 : !int32_t
      }
    // CHECK-NEXT: break;
    // CHECK: }
    }

    // CHECK-NEXT: switch (3) {
    clift.switch {
      %0 = clift.imm 3 : !int32_t
      clift.yield %0 : !int32_t
    // CHECK-NEXT: case 0x3:
    } case 3 {
      // CHECK-NEXT: 4;
      clift.expr {
        %1 = clift.imm 4 : !int32_t
        clift.yield %1 : !int32_t
      }
    // CHECK-NEXT: break;
    // CHECK-NEXT: default:
    } default {
      // CHECK-NEXT: 5;
      clift.expr {
        %2 = clift.imm 5 : !int32_t
        clift.yield %2 : !int32_t
      }
    // CHECK-NEXT: break;
    // CHECK: }
    } attributes {clift.radix = 16 : ui32}
  }
  // CHECK-NEXT: }
}
