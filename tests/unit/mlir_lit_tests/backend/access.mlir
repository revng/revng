//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s --emit-c="tagless model=%S/model.yml" -o /dev/null | FileCheck %s

!void = !clift.primitive<VoidKind 0>

!int32_t = !clift.primitive<SignedKind 4>

!s = !clift.defined<#clift.struct<
  unique_handle = "/model-type/2002",
  name = "",
  size = 8,
  fields = [
    <
      name = "",
      offset = 0,
      type = !int32_t
    >,
    <
      name = "",
      offset = 4,
      type = !int32_t
    >
  ]>>
!s$p = !clift.pointer<pointer_size = 8, pointee_type = !s>

!u = !clift.defined<#clift.union<
  unique_handle = "/model-type/2003",
  name = "",
  fields = [
    <
      name = "",
      offset = 0,
      type = !int32_t
    >,
    <
      name = "",
      offset = 0,
      type = !int32_t
    >
  ]>>
!u$p = !clift.pointer<pointer_size = 8, pointee_type = !u>

!f = !clift.defined<#clift.function<
  unique_handle = "/model-type/1001",
  name = "",
  return_type = !void,
  argument_types = []>>

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
