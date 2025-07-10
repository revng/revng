//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngpipe clift-legalization %S/model.yml %s /dev/stdout | %revngcliftopt | FileCheck %s

!void = !clift.primitive<void 0>
!int32_t = !clift.primitive<signed 4>

!f = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" : !void()
>

module attributes {clift.module} {
  clift.func @f<!f>() attributes {
    handle = "/function/0x40001001:Code_x86_64"
  } {
    clift.expr {
      %0 = clift.undef : !clift.ptr<4 to !int32_t>
      // CHECK: %1 = clift.cast<extend> %0 : !clift.ptr<4 to !int32_t> -> !clift.ptr<8 to !int32_t>
      %1 = clift.indirection %0 : !clift.ptr<4 to !int32_t>
      // CHECK: %2 = clift.indirection %1 : !clift.ptr<8 to !int32_t>
      // CHECK: clift.yield %2 : !int32_t
      clift.yield %1 : !int32_t
    }
  }
}
