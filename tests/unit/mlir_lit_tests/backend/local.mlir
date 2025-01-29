//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s --emit-c="tagless model=%S/model.yml" -o /dev/null | FileCheck %s

!void = !clift.primitive<VoidKind 0>
!int32_t = !clift.primitive<SignedKind 4>

!f = !clift.defined<#clift.function<
  unique_handle = "/model-type/1001",
  name = "",
  return_type = !void,
  argument_types = []>>

!my_enum = !clift.defined<#clift.enum<
  unique_handle = "/model-type/2001",
  name = "",
  underlying_type = !int32_t,
  fields = [
    <
      raw_value = 0,
      name = ""
    >
  ]>>

!my_struct = !clift.defined<#clift.struct<
  unique_handle = "/model-type/2002",
  name = "",
  size = 4,
  fields = [
    <
      offset = 0,
      name = "",
      type = !int32_t
    >
  ]>>

!my_union = !clift.defined<#clift.union<
  unique_handle = "/model-type/2003",
  name = "",
  fields = [
    <
      offset = 0,
      name = "",
      type = !int32_t
    >
  ]>>

clift.module {
  // CHECK: void fun_0x40001001(void) {
  clift.func @f<!f>() attributes {
    unique_handle = "/function/0x40001001:Code_x86_64"
  } {
    // CHECK: enum my_enum _var_0;
    %e = clift.local !my_enum "e"

    // CHECK: struct my_struct _var_1;
    %s = clift.local !my_struct "s"

    // CHECK: union my_union _var_2;
    %u = clift.local !my_union "u"

    // CHECK: int32_t _var_3 = 42;
    %i = clift.local !int32_t "i" = {
      %42 = clift.imm 42 : !int32_t
      clift.yield %42 : !int32_t
    }

    // _var_0;
    clift.expr {
        clift.yield %e : !my_enum
    }

    // _var_1;
    clift.expr {
        clift.yield %s : !my_struct
    }

    // _var_2;
    clift.expr {
        clift.yield %u : !my_union
    }

    // _var_3;
    clift.expr {
        clift.yield %i : !int32_t
    }
  }
  // CHECK: }
}
