//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s --c-legalization | FileCheck %s

!void = !clift.primitive<void 0>
!int16_t = !clift.primitive<signed 2>
!int32_t = !clift.primitive<signed 4>

!f = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" : !void()
>

module attributes {clift.module} {
  clift.func @f<!f>() attributes {
    handle = "/function/0x40001001:Code_x86_64"
  } {
    clift.expr {
      %0 = clift.imm 0 : !int32_t
      %1 = clift.imm 0 : !int32_t
      // CHECK: %2 = clift.eq %0, %1 : !int32_t -> !int32_t
      %2 = clift.eq %0, %1 : !int32_t -> !int16_t
      // CHECK: %3 = clift.cast<truncate> %2 : !int32_t -> !int16_t
      // CHECK: clift.yield %3 : !int16_t
      clift.yield %2 : !int16_t
    }
  }
}
