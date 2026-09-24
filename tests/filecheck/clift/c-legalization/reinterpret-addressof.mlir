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
    // CHECK: %0 = clift.local : !int32_t
    %0 = clift.local : !int32_t
    // CHECK: clift.expr {
    clift.expr {
      // CHECK: %1 = clift.addressof %0 : !clift.ptr<8 to !int32_t>
      // CHECK: %2 = clift.bitcast %1 : !clift.ptr<8 to !int32_t> -> !clift.ptr<8 to !uint32_t>
      %1 = clift.reinterpret %0 : !int32_t -> !uint32_t
      %2 = clift.addressof %1 : !clift.ptr<8 to !uint32_t>
      // CHECK: clift.yield %2 : !clift.ptr<8 to !uint32_t>
      clift.yield %2 : !clift.ptr<8 to !uint32_t>
    // CHECK: }
    }
  }
}
