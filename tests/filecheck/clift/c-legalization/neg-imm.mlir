//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s --c-legalization | FileCheck %s

!void = !clift.void
!int16_t = !clift.int<signed 2>
!int32_t = !clift.int<signed 4>

!f = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" : !void()
>

module attributes {clift.module} {
  clift.func @f<!f>() attributes {
    handle = "/function/0x40001001:Code_x86_64"
  } {
    clift.expr {
      // CHECK: %0 = clift.imm 0 : !int32_t
      %0 = clift.imm 0 : !int16_t
      // CHECK: %1 = clift.neg %0 : !int32_t
      %1 = clift.neg %0 : !int16_t
      // CHECK: clift.yield %1 : !int32_t
      clift.yield %1 : !int16_t
    }
  }
}
