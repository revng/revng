//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt --emit-c %s | FileCheck %s
// RUN: %revngcliftopt --emit-c=ptml %s -o /dev/null | %revngptml | FileCheck %s

!void = !clift.primitive<void 0>
!int32_t = !clift.primitive<signed 4>

!f = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" : !void()
>

module attributes {clift.module} {
  clift.global @seg_0x40002001 : !int32_t attributes {
    handle = "/segment/0x40002001:Generic64-4"
  }

  // CHECK: void fun_0x40001001(void) {
  clift.func @fun_0x40001001<!f>() attributes {
    handle = "/function/0x40001001:Code_x86_64"
  } {
    // CHECK: seg_0x40002001;
    clift.expr {
      %y = clift.use @seg_0x40002001 : !int32_t
      clift.yield %y : !int32_t
    }
  }
  // CHECK: }
}
