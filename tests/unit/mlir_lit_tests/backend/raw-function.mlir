//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s --emit-c="tagless model=%S/model.yml" -o /dev/null | FileCheck %s

!int32_t = !clift.primitive<SignedKind 4>

!f_args = !clift.defined<#clift.struct<
  unique_handle = "/model-type/2004",
  name = "",
  size = 4,
  fields = [
    <
      offset = 0,
      name = "",
      type = !int32_t
    >
  ]>>

!f = !clift.defined<#clift.function<
  unique_handle = "/model-type/1003",
  name = "",
  return_type = !int32_t,
  argument_types = [!int32_t, !f_args]>>

clift.module {
  // CHECK: int32_t fun_0x40001003(int32_t rcx _REG(rcx_x86_64), args_1003 _stack_arguments _STACK) {
  clift.func @f<!f>(%arg0 : !int32_t, %arg1 : !f_args) attributes {
    unique_handle = "/function/0x40001003:Code_x86_64"
  } {
    // CHECK: return _stack_arguments.a + rcx;
    clift.return {
      %a = clift.access<0> %arg1 : !f_args -> !int32_t
      %r = clift.add %a, %arg0 : !int32_t
      clift.yield %r : !int32_t
    }
  }
  // CHECK: }
}
