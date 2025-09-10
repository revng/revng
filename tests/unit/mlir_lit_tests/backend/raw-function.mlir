//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt --emit-c %s | FileCheck %s
// RUN: %revngcliftopt --emit-c=ptml %s -o /dev/null | %revngptml | FileCheck %s

!int32_t = !clift.primitive<signed 4>

!f_args = !clift.struct<
  "/type-definition/2004-StructDefinition" as "args_1003" : size(4) {
    "/struct-field/2004-StructDefinition/0" as "a" : offset(0) !int32_t
  }
>

!f = !clift.func<
  "/type-definition/1003-RawFunctionDefinition" : !int32_t(!int32_t, !f_args)
>

module attributes {clift.module} {
  // CHECK: int32_t fun_0x40001003(int32_t rcx, args_1003 stack_arguments) {
  clift.func @fun_0x40001003<!f>(%arg0 : !int32_t { clift.handle = "/raw-argument/1003-RawFunctionDefinition/rcx_x86_64",
                                                    clift.name = "rcx" },
                                 %arg1 : !f_args { clift.handle = "/raw-stack-arguments/1003-RawFunctionDefinition",
                                                   clift.name = "stack_arguments" }) attributes {
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
