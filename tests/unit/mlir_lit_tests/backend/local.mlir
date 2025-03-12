//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s --emit-c="tagless model=%S/model.yml" -o /dev/null | FileCheck %s

!void = !clift.primitive<void 0>
!int32_t = !clift.primitive<signed 4>

!f = !clift.defined<#clift.func<
  "/model-type/1001" : !void()>>

!my_enum = !clift.defined<#clift.enum<
  "/model-type/2001" : !int32_t {
    0
  }>>

!my_struct = !clift.defined<#clift.struct<
  "/model-type/2002" : size(4) {
    offset(0) : !int32_t
  }>>

!my_union = !clift.defined<#clift.union<
  "/model-type/2003" : {
    !int32_t
  }>>

!my_pair = !clift.defined<#clift.struct<
  "/model-type/2006" : size(8) {
    offset(0) : !int32_t,
    offset(4) : !int32_t
  }>>

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

    // CHECK: struct my_pair _var_4 = {
    // CHECK:   1,
    // CHECK:   2
    // CHECK: };
    %p = clift.local !my_pair "p" = {
      %1 = clift.imm 1 : !int32_t
      %2 = clift.imm 2 : !int32_t
      %r = clift.aggregate(%1, %2) : !my_pair
      clift.yield %r : !my_pair
    }

    // CHECK: _var_0;
    clift.expr {
        clift.yield %e : !my_enum
    }

    // CHECK: _var_1;
    clift.expr {
        clift.yield %s : !my_struct
    }

    // CHECK: _var_2;
    clift.expr {
        clift.yield %u : !my_union
    }

    // CHECK: _var_3;
    clift.expr {
        clift.yield %i : !int32_t
    }

    // CHECK: _var_4;
    clift.expr {
        clift.yield %p : !my_pair
    }
  }
  // CHECK: }
}
