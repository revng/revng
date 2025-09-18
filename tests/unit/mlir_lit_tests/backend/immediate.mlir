//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngpipe emit-c %S/model.yml %s <(tar -czT /dev/null) /dev/stdout | tar -zxO

!void = !clift.primitive<void 0>

!int32_t = !clift.primitive<signed 4>
!uint32_t = !clift.primitive<unsigned 4>

!f = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" : !void()
>

!my_enum = !clift.enum<
  "/type-definition/2001-EnumDefinition" as "my_enum" : !int32_t {
    "/enum-entry/2001-EnumDefinition/0" as "my_enum_0" : 0
  }
>

module attributes {clift.module} {
  // CHECK: void fun_0x40001001(void) {
  clift.func @fun_0x40001001<!f>() attributes {
    handle = "/function/0x40001001:Code_x86_64"
  } {
    // CHECK: 0;
    clift.expr {
      %i = clift.imm 0 : !int32_t
      clift.yield %i : !int32_t
    }

    // CHECK: 0U;
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
