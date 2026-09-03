//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt --emit-c %s -o /dev/null | FileCheck %s
// RUN: %root/bin/revng clift-opt --emit-c=ptml %s -o /dev/null | %root/bin/revng ptml | FileCheck %s

!void = !clift.void
!int64_t = !clift.int<signed 8>

!f = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" : !void()
>

module attributes {clift.module} {
  // CHECK: void fun_0x40001001(void) {
  clift.func @fun_0x40001001<!f>() attributes {
    handle = "/function/0x40001001:Code_x86_64"
  } {
    // CHECK: NULL;
    clift.expr {
      %0 = clift.null : !clift.ptr<8 to !clift.void>
      %1 = clift.bitcast %0 : !clift.ptr<8 to !clift.void> -> !clift.ptr<8 to !int64_t>
      clift.yield %1 : !clift.ptr<8 to !int64_t>
    }
  }
  // CHECK: }
}
