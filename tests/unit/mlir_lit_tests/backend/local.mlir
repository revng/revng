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
  "/type-definition/2001-EnumDefinition" : !int32_t {
    "/enum-entry/2001-EnumDefinition/0" : 0
  }
>

!my_struct = !clift.struct<
  "/type-definition/2002-StructDefinition" : size(8) {
    "/struct-field/2002-StructDefinition/0" : offset(0) !int32_t,
    "/struct-field/2002-StructDefinition/1" : offset(4) !int32_t
  }
>

!my_union = !clift.union<
  "/type-definition/2003-UnionDefinition" : {
    "/union-field/2003-UnionDefinition/0" : !int32_t
  }
>

!my_pair = !clift.struct<
  "/type-definition/2006-StructDefinition" : size(8) {
    "/struct-field/2006-StructDefinition/0" : offset(0) !int32_t,
    "/struct-field/2006-StructDefinition/4" : offset(4) !int32_t
  }
>

module attributes {clift.module} {
  // CHECK: void fun_0x40001001(void) {
  clift.func @f<!f>() attributes {
    handle = "/function/0x40001001:Code_x86_64"
  } {
    // CHECK: my_enum var_0;
    %e = clift.local : !my_enum

    // CHECK: my_struct var_1;
    %s = clift.local : !my_struct

    // CHECK: my_union var_2;
    %u = clift.local : !my_union

    // CHECK: int32_t var_3 = 42;
    %i = clift.local : !int32_t = {
      %42 = clift.imm 42 : !int32_t
      clift.yield %42 : !int32_t
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
