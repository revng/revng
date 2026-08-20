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
    // CHECK: %0 = clift.local : !int16_t
    %0 = clift.local : !int16_t
    clift.expr {
      // CHECK: %1 = clift.sext %0 : !int16_t -> !int32_t
      // CHECK: %2 = clift.sext %0 : !int16_t -> !int32_t
      // CHECK: %3 = clift.add %1, %2 : !int32_t
      %1 = clift.add %0, %0 : !int16_t
      // CHECK: %4 = clift.truncate %3 : !int32_t -> !int16_t
      // CHECK: %5 = clift.sext %4 : !int16_t -> !int32_t
      // CHECK: %6 = clift.sext %0 : !int16_t -> !int32_t
      // CHECK: %7 = clift.add %5, %6 : !int32_t
      %2 = clift.add %1, %0 : !int16_t
      // CHECK: clift.yield %7 : !int32_t
      clift.yield %2 : !int16_t
    }
  }
}
