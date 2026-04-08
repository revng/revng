//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s --c-legalization | FileCheck %s

!void = !clift.void
!int8_t = !clift.int<signed 1>

!f = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" : !void()
>

module attributes {clift.module} {
  clift.func @f<!f>() attributes {
    handle = "/function/0x40001001:Code_x86_64"
  } {
    clift.expr {
      // CHECK: %0 = clift.imm 42 : !int32_t
      %0 = clift.imm 42 : !int8_t
      // CHECK: %1 = clift.truncate %0 : !int32_t -> !int8_t
      // CHECK: clift.yield %1 : !int8_t
      clift.yield %0 : !int8_t
    }
  }
}
