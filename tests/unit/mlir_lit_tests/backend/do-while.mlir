//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt --emit-c %s | FileCheck %s
// RUN: %revngcliftopt --emit-c=ptml %s -o /dev/null | %revngptml | FileCheck %s

!void = !clift.primitive<void 0>
!int32_t = !clift.primitive<signed 4>

!f = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" : !void()
>

module attributes {clift.module} {
  // CHECK: void fun_0x40001001(void) {
  clift.func @fun_0x40001001<!f>() attributes {
    handle = "/function/0x40001001:Code_x86_64"
  } {
    // CHECK: do
    clift.do_while body {
      // CHECK: 0;
      clift.expr {
        %0 = clift.imm 0 : !int32_t
        clift.yield %0 : !int32_t
      }
    // CHECK: while (1);
    } cond {
      %1 = clift.imm 1 : !int32_t
      clift.yield %1 : !int32_t
    }

    // CHECK: do {
    clift.do_while body {
      // CHECK: 2;
      clift.expr {
        %2 = clift.imm 2 : !int32_t
        clift.yield %2 : !int32_t
      }
      // CHECK: 3;
      clift.expr {
        %3 = clift.imm 3 : !int32_t
        clift.yield %3 : !int32_t
      }
    // CHECK: } while (4);
    } cond {
      %4 = clift.imm 4 : !int32_t
      clift.yield %4 : !int32_t
    }
  }
  // CHECK: }
}
