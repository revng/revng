//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s --emit-c="tagless model=%S/model.yml" -o /dev/null | FileCheck %s

!void = !clift.primitive<void 0>
!int32_t = !clift.primitive<signed 4>

!f = !clift.defined<#clift.func<
  "/model-type/1001" : !void()>>

clift.module {
  clift.global !int32_t @g {
    handle = "/segment/0x40002001:Generic64-4"
  }

  // CHECK: void fun_0x40001001(void) {
  clift.func @f<!f>() attributes {
    handle = "/function/0x40001001:Code_x86_64"
  } {
    // CHECK: seg_0x40002001;
    clift.expr {
      %y = clift.use @g : !int32_t
      clift.yield %y : !int32_t
    }
  }
  // CHECK: }
}
