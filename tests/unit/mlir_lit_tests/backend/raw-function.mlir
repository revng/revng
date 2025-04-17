//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s --emit-c="tagless model=%S/model.yml" -o /dev/null | FileCheck %s

!int32_t = !clift.primitive<signed 4>

!f_args = !clift.struct<
  "/type-definition/2004-StructDefinition" : size(4) {
    offset(0) : !int32_t
  }
>

!f = !clift.func<
  "/type-definition/1003-CABIFunctionDefinition" : !int32_t(!int32_t, !f_args)
>

module attributes {clift.module} {
  // CHECK: int32_t fun_0x40001003(int32_t rcx _REG(rcx_x86_64), args_1003 stack_arguments _STACK) {
  clift.func @f<!f>(%arg0 : !int32_t, %arg1 : !f_args) attributes {
    handle = "/function/0x40001003:Code_x86_64"
  } {
    // CHECK: return stack_arguments.a + rcx;
    clift.return {
      %a = clift.access<0> %arg1 : !f_args -> !int32_t
      %r = clift.add %a, %arg0 : !int32_t
      clift.yield %r : !int32_t
    }
  }
  // CHECK: }
}
