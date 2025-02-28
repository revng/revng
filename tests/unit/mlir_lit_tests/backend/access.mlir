//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s --emit-c="tagless model=%S/model.yml" -o /dev/null | FileCheck %s

!void = !clift.primitive<void 0>

!int32_t = !clift.primitive<signed 4>

!s = !clift.defined<#clift.struct<
  "/model-type/2002" : size(8) {
    offset(0) : !int32_t,
    offset(4) : !int32_t
  }>>
!s$p = !clift.ptr<8 to !s>

!u = !clift.defined<#clift.union<"/model-type/2003" : {
    !int32_t,
    !int32_t
  }>>
!u$p = !clift.ptr<8 to !u>

!f = !clift.defined<#clift.func<
  "/model-type/1001" : !void()>>

clift.module {
  // CHECK: void fun_0x40001001(void) {
  clift.func @f<!f>() attributes {
    unique_handle = "/function/0x40001001:Code_x86_64"
  } {
    %s = clift.local !s$p "s"
    %u = clift.local !u$p "u"

    // CHECK: _var_0->x;
    clift.expr {
      %a = clift.access<indirect 0> %s : !s$p -> !int32_t
      clift.yield %a : !int32_t
    }

    // CHECK: _var_1->x;
    clift.expr {
      %a = clift.access<indirect 0> %u : !u$p -> !int32_t
      clift.yield %a : !int32_t
    }

    // CHECK: _var_0->y;
    clift.expr {
      %a = clift.access<indirect 1> %s : !s$p -> !int32_t
      clift.yield %a : !int32_t
    }

    // CHECK: _var_1->y;
    clift.expr {
      %a = clift.access<indirect 1> %u : !u$p -> !int32_t
      clift.yield %a : !int32_t
    }

    // CHECK: (*_var_0).x;
    clift.expr {
      %v = clift.indirection %s : !s$p
      %a = clift.access<0> %v : !s -> !int32_t
      clift.yield %a : !int32_t
    }

    // CHECK: (*_var_1).x;
    clift.expr {
      %v = clift.indirection %u : !u$p
      %a = clift.access<0> %v : !u -> !int32_t
      clift.yield %a : !int32_t
    }
  }
  // CHECK: }
}
