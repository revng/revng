//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s --c-legalization | FileCheck %s

!void = !clift.void
!int32_t = !clift.int<signed 4>

!f = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" : !void()
>

!s = !clift.struct<
  "/type-definition/1002-StructDefinition" as "s" : size(4) {
    "/struct-field/1002-StructDefinition/0" : offset(0) !int32_t
  }
>

module attributes {clift.module} {
  clift.func @f<!f>() attributes {
    handle = "/function/0x40001001:Code_x86_64"
  } {
    // CHECK: %0 = clift.local : !int32_t
    %0 = clift.local : !int32_t
    // CHECK: clift.expr {
    clift.expr {
      // CHECK: %1 = clift.addressof %0 : !clift.ptr<8 to !int32_t>
      // CHECK: %2 = clift.bitcast %1 : !clift.ptr<8 to !int32_t> -> !clift.ptr<8 to !s>
      // CHECK: %3 = clift.ptr_access<0> %2 : !clift.ptr<8 to !s> -> !int32_t
      %1 = clift.reinterpret %0 : !int32_t -> !s
      %2 = clift.access<0> %1 : !s -> !int32_t
      // CHECK: clift.yield %3 : !int32_t
      clift.yield %2 : !int32_t
    // CHECK: }
    }
  }
}
