//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s --emit-c="tagless model=%S/model.yml" -o /dev/null | FileCheck %s

!void = !clift.primitive<void 0>
!int32_t = !clift.primitive<signed 4>

!s = !clift.defined<#clift.struct<"/model-type/2002" : size(8) {
    offset(0) : !int32_t,
    offset(4) : !int32_t
  }>>

!f = !clift.defined<
  #clift.func<"/model-type/1001" : !void()>>

clift.module {
  // CHECK: void fun_0x40001001(void) {
  clift.func @f<!f>() attributes {
    handle = "/function/0x40001001:Code_x86_64"
  } {
    // CHECK: (struct my_struct){0, 1};
    clift.expr {
      %0 = clift.imm 0 : !int32_t
      %1 = clift.imm 1 : !int32_t
      %s = clift.aggregate(%0, %1) : !s
      clift.yield %s : !s
    }
  }
  // CHECK: }
}
