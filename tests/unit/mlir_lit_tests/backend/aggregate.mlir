//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngpipe emit-c %S/model.yml %s <(tar -czT /dev/null) /dev/stdout | tar -zxO

!void = !clift.primitive<void 0>
!int32_t = !clift.primitive<signed 4>

!s = !clift.struct<
  "/type-definition/2002-StructDefinition" : size(8) {
    "/struct-field/2002-StructDefinition/0" : offset(0) !int32_t,
    "/struct-field/2002-StructDefinition/4" : offset(4) !int32_t
  }
>

!f = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" : !void()
>

module attributes {clift.module} {
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
