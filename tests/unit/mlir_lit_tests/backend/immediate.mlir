//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s --emit-c="tagless model=%S/model.yml" -o /dev/null | FileCheck %s

!void = !clift.primitive<void 0>

!int32_t = !clift.primitive<signed 4>
!uint32_t = !clift.primitive<unsigned 4>

!f = !clift.defined<#clift.function<
  unique_handle = "/model-type/1001",
  name = "",
  return_type = !void,
  argument_types = []>>

!my_enum = !clift.defined<#clift.enum<"/model-type/2001" : !int32_t { 0 }>>

clift.module {
  // CHECK: void fun_0x40001001(void) {
  clift.func @f<!f>() attributes {
    unique_handle = "/function/0x40001001:Code_x86_64"
  } {
    // CHECK: 0;
    clift.expr {
      %i = clift.imm 0 : !int32_t
      clift.yield %i : !int32_t
    }

    // CHECK: 0u;
    clift.expr {
      %u = clift.imm 0 : !uint32_t
      clift.yield %u : !uint32_t
    }

    // CHECK: my_enum_0;
    clift.expr {
      %e = clift.imm 0 : !my_enum
      clift.yield %e : !my_enum
    }
  }
  // CHECK: }
}
