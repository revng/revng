//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s --emit-c="tagless model=%S/model.yml" -o /dev/null | FileCheck %s

!void = !clift.primitive<VoidKind 0>

!f = !clift.defined<#clift.function<
  unique_handle = "/model-type/1005",
  name = "",
  return_type = !void,
  argument_types = []>>

clift.module {
  // CHECK: void fun_0x40001005(void) {
  clift.func @f<!f>() attributes {
    unique_handle = "/function/0x40001005:Code_x86_64"
  } {
    // CHECK: struct {{.*}}frame_1005 {
    // CHECK: int32_t a;
    // CHECK: };

    ^0: // Empty block to force function definition.
  }
  // CHECK: }
}
