//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s --c-legalization | FileCheck %s

!void = !clift.void

!int32_t = !clift.int<signed 4>
!uint32_t = !clift.int<unsigned 4>

!int64_t = !clift.int<signed 8>
!uint64_t = !clift.int<unsigned 8>

!f = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" : !void()
>

module attributes {clift.module} {
  clift.func @f<!f>() attributes {
    handle = "/function/0x40001001:Code_x86_64"
  } {
    clift.expr {
      // CHECK: %0 = clift.imm 42 : !int32_t
      %0 = clift.imm 42 : !int32_t
      // CHECK: clift.yield %0 : !int32_t
      clift.yield %0 : !int32_t
    }

    clift.expr {
      // CHECK: %0 = clift.imm 42 : !uint32_t
      %0 = clift.imm 42 : !uint32_t
      // CHECK: clift.yield %0 : !uint32_t
      clift.yield %0 : !uint32_t
    }

    clift.expr {
      // CHECK: %0 = clift.imm 42 : !int64_t
      %0 = clift.imm 42 : !int64_t
      // CHECK: clift.yield %0 : !int64_t
      clift.yield %0 : !int64_t
    }

    clift.expr {
      // CHECK: %0 = clift.imm 42 : !uint64_t
      %0 = clift.imm 42 : !uint64_t
      // CHECK: clift.yield %0 : !uint64_t
      clift.yield %0 : !uint64_t
    }
  }
}
