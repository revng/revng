//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s --emit-c="tagless model=%S/model.yml" -o /dev/null | FileCheck %s

!int32_t = !clift.primitive<signed 4>

!f = !clift.defined<#clift.function<
  unique_handle = "/model-type/1001",
  name = "",
  return_type = !clift.primitive<void 0>,
  argument_types = []>>

clift.module {
  // CHECK: void fun_0x40001001(void) {
  clift.func @f<!f>() attributes {
    unique_handle = "/function/0x40001001:Code_x86_64"
  } {
    // CHECK: 0 ? 1 : 2;
    clift.expr {
      %0 = clift.imm 0 : !int32_t
      %1 = clift.imm 1 : !int32_t
      %2 = clift.imm 2 : !int32_t
      %r = clift.ternary %0, %1, %2 : (!int32_t, !int32_t)
      clift.yield %r : !int32_t
    }
  }
  // CHECK: }
}
