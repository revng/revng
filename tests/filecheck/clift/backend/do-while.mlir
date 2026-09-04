//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt --emit-c %s -o /dev/null | FileCheck %s
// RUN: %root/bin/revng clift-opt --emit-c=ptml %s -o /dev/null | %root/bin/revng ptml | FileCheck %s

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
    // CHECK: do
    clift.do_while body {
      // CHECK: 0;
      clift.expr {
        %0 = clift.imm 0 : !int32_t
        clift.yield %0 : !int32_t
      }
    // CHECK: while ((bool) 1);
    } cond {
      %1 = clift.imm 1 : !int32_t
      %2 = clift.test %1 : !int32_t
      clift.yield %2 : !clift.bool
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
    // CHECK: } while ((bool) 4);
    } cond {
      %4 = clift.imm 4 : !int32_t
      %5 = clift.test %4 : !int32_t
      clift.yield %5 : !clift.bool
    }
  }
  // CHECK: }
}
