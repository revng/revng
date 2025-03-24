//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s --emit-c="tagless model=%S/model.yml" -o /dev/null | FileCheck %s

!int32_t = !clift.primitive<signed 4>

!s = !clift.defined<#clift.struct<
  unique_handle = "/model-type/2002",
  name = "",
  size = 8,
  fields = [
    <
      name = "",
      offset = 0,
      type = !int32_t
    >,
    <
      name = "",
      offset = 4,
      type = !int32_t
    >
  ]>>

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
