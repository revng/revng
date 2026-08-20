//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s --c-legalization | FileCheck %s

!void = !clift.void
!uint32_t = !clift.int<unsigned 4>
!uint64_t = !clift.int<unsigned 8>

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
      // CHECK: %1 = clift.bitcast %0 : !uint32_t -> !int32_t
      // CHECK: %2 = clift.sext %1 : !int32_t -> !uint64_t
      %1 = clift.sext %0 : !uint32_t -> !uint64_t
      // CHECK: clift.yield %2 : !uint64_t
      clift.yield %1 : !uint64_t
    }
  }
}
