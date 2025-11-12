//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s --c-legalization | FileCheck %s

!void = !clift.primitive<void 0>
!int8_t = !clift.primitive<signed 1>
!int32_t = !clift.primitive<signed 4>

!f = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" : !void()
>

module attributes {clift.module} {
  clift.func @f<!f>() attributes {
    handle = "/function/0x40001001:Code_x86_64"
  } {
    // CHECK: %0 = clift.local : !int32_t
    %0 = clift.local : !int32_t
    // CHECK: %1 = clift.local : !int8_t
    %1 = clift.local : !int8_t
    clift.expr {
      // CHECK: %2 = clift.eq %0, %0 : !int32_t -> !int32_t
      %2 = clift.eq %0, %0 : !int32_t -> !int8_t
      // CHECK: %3 = clift.cast<truncate> %2 : !int32_t -> !int8_t
      // CHECK: %4 = clift.eq %1, %3 : !int8_t -> !int32_t
      %3 = clift.eq %1, %2 : !int8_t -> !int8_t
      // CHECK: clift.yield %4 : !int32_t
      clift.yield %3 : !int8_t
    }
  }
}
