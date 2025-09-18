//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngpipe emit-c %S/model.yml %s <(tar -czT /dev/null) /dev/stdout | tar -zxO

!void = !clift.primitive<void 0>
!int32_t = !clift.primitive<signed 4>

!f = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" : !void()
>

!my_enum = !clift.enum<
  "/type-definition/2001-EnumDefinition" as "my_enum" : !int32_t {
    "/enum-entry/2001-EnumDefinition/0" as "my_enum_0" : 0
  }
>

!my_struct = !clift.struct<
  "/type-definition/2002-StructDefinition" as "my_struct" : size(8) {
    "/struct-field/2002-StructDefinition/0" as "x" : offset(0) !int32_t,
    "/struct-field/2002-StructDefinition/1" as "y" : offset(4) !int32_t
  }
>

!my_union = !clift.union<
  "/type-definition/2003-UnionDefinition" as "my_union" : {
    "/union-field/2003-UnionDefinition/0" as "x" : !int32_t
  }
>

!my_pair = !clift.struct<
  "/type-definition/2006-StructDefinition" as "my_pair" : size(8) {
    "/struct-field/2006-StructDefinition/0" as "a" : offset(0) !int32_t,
    "/struct-field/2006-StructDefinition/4" as "b" : offset(4) !int32_t
  }
>

module attributes {clift.module} {
  // CHECK: void fun_0x40001001(void) {
  clift.func @fun_0x40001001<!f>() attributes {
    handle = "/function/0x40001001:Code_x86_64"
  } {
    // CHECK: my_enum var_0;
    %e = clift.local : !my_enum attributes {
      handle = "/local-variable/0x40001001:Code_x86_64/0",
      clift.name = "var_0"
    }

    // CHECK: my_struct var_1;
    %s = clift.local : !my_struct attributes {
      handle = "/local-variable/0x40001001:Code_x86_64/1",
      clift.name = "var_1"
    }

    // CHECK: my_union var_2;
    %u = clift.local : !my_union attributes {
      handle = "/local-variable/0x40001001:Code_x86_64/2",
      clift.name = "var_2"
    }

    // CHECK: int32_t var_3 = 42;
    %i = clift.local : !int32_t = {
      %42 = clift.imm 42 : !int32_t
      clift.yield %42 : !int32_t
    } attributes {
      handle = "/local-variable/0x40001001:Code_x86_64/3",
      clift.name = "var_3"
    }

    // CHECK: my_pair var_4 = {
    // CHECK:   1,
    // CHECK:   2
    // CHECK: };
    %p = clift.local : !my_pair = {
      %1 = clift.imm 1 : !int32_t
      %2 = clift.imm 2 : !int32_t
      %r = clift.aggregate(%1, %2) : !my_pair
      clift.yield %r : !my_pair
    } attributes {
      handle = "/local-variable/0x40001001:Code_x86_64/4",
      clift.name = "var_4"
    }

    // CHECK: var_0;
    clift.expr {
        clift.yield %e : !my_enum
    }

    // CHECK: var_1;
    clift.expr {
        clift.yield %s : !my_struct
    }

    // CHECK: var_2;
    clift.expr {
        clift.yield %u : !my_union
    }

    // CHECK: var_3;
    clift.expr {
        clift.yield %i : !int32_t
    }

    // CHECK: var_4;
    clift.expr {
        clift.yield %p : !my_pair
    }
  }
  // CHECK: }
}
