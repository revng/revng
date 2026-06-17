//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s --c-legalization | FileCheck %s

!void = !clift.void
!int32_t = !clift.int<signed 4>
!uint32_t = !clift.int<unsigned 4>

!f = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" : !void()
>

module attributes {clift.module} {
  clift.func @f<!f>() attributes {
    handle = "/function/0x40001001:Code_x86_64"
  } {
    clift.expr {
      // CHECK: %0 = clift.undef : !uint32_t
      %0 = clift.undef : !uint32_t
      // CHECK: %1 = clift.undef : !uint32_t
      %1 = clift.undef : !uint32_t
      // CHECK: %2 = clift.bitcast %0 : !uint32_t -> !int32_t
      // CHECK: %3 = clift.sar %2, %1 : (!int32_t, !uint32_t)
      %2 = clift.sar %0, %1 : !uint32_t
      // CHECK: clift.yield %3 : !int32_t
      clift.yield %2 : !uint32_t
    }
  }
}
