//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngpipe clift-legalization %S/model.yml %s /dev/stdout | %revngcliftopt | FileCheck %s

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
    %0 = clift.local !int16_t
    clift.expr {
      // CHECK: %1 = clift.cast<extend> %0 : !int16_t -> !int32_t
      // CHECK: %2 = clift.cast<extend> %0 : !int16_t -> !int32_t
      // CHECK: %3 = clift.add %1, %2 : !int32_t
      %1 = clift.add %0, %0 : !int16_t
      // CHECK: %4 = clift.cast<truncate> %3 : !int32_t -> !int16_t
      clift.yield %1 : !int16_t
    }
  }
}
