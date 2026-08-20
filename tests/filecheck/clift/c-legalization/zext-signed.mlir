//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s --c-legalization | FileCheck %s

!void = !clift.void
!int32_t = !clift.int<signed 4>
!int64_t = !clift.int<signed 8>

!f = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" : !void()
>

module attributes {clift.module} {
  clift.func @f<!f>() attributes {
    handle = "/function/0x40001001:Code_x86_64"
  } {
    clift.expr {
      // CHECK: %0 = clift.undef : !int32_t
      %0 = clift.undef : !int32_t
      // CHECK: %1 = clift.bitcast %0 : !int32_t -> !uint32_t
      // CHECK: %2 = clift.zext %1 : !uint32_t -> !int64_t
      %1 = clift.zext %0 : !int32_t -> !int64_t
      // CHECK: clift.yield %2 : !int64_t
      clift.yield %1 : !int64_t
    }
  }
}
