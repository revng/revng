//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt --emit-c %s -o /dev/null | FileCheck %s
// RUN: %root/bin/revng clift-opt --emit-c=ptml %s -o /dev/null | %root/bin/revng ptml | FileCheck %s

!void = !clift.void

!int32_t = !clift.int<signed 4>
!int512_t = !clift.int<signed 64>

!f = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" : !void()
>

module attributes {clift.module} {
  // CHECK: void fun_0x40001001(void) {
  clift.func @fun_0x40001001<!f>() attributes {
    handle = "/function/0x40001001:Code_x86_64"
  } {
    // CHECK: rr_truncate(int32_t, rr_add(rr_imm(int512_t, 0), rr_imm(int512_t, 1)));
    clift.expr {
      %0 = clift.imm 0 {clift.intrinsic} : !int512_t
      %1 = clift.imm 1 {clift.intrinsic} : !int512_t
      %2 = clift.add %0, %1 {clift.intrinsic} : !int512_t
      %3 = clift.truncate %2 {clift.intrinsic} : !int512_t -> !int32_t
      clift.yield %3 : !int32_t
    }
  }
  // CHECK: }
}
