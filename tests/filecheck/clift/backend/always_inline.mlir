//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt --emit-c %s -o /dev/null | FileCheck %s
// RUN: %root/bin/revng clift-opt --emit-c=ptml %s -o /dev/null | %root/bin/revng ptml | FileCheck %s

!void = !clift.void
!int32_t = !clift.int<signed 4>

!f = !clift.func<"/type-definition/1004-CABIFunctionDefinition" : !void()>

module attributes { clift.module } {
  // CHECK: _ALWAYS_INLINE
  // CHECK: void fun_0x40001004(void)
  // CHECK: {
  clift.func @fun_0x40001004<!f>() attributes {
    handle = "/function/0x40001004:Code_x86_64",
    always_inline
  } {
    clift.expr {
      %0 = clift.imm 0 : !int32_t
      clift.yield %0 : !int32_t
    }
  }
}
