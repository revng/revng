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
!s$p = !clift.ptr<8 to !s>

!u = !clift.union<
  "/type-definition/2003-UnionDefinition" : {
    "/union-field/2003-UnionDefinition/0" : !int32_t,
    "/union-field/2003-UnionDefinition/1" : !int32_t
  }
>
!u$p = !clift.ptr<8 to !u>

!f = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" : !void()
>

module attributes {clift.module} {
  // CHECK: void fun_0x40001001(void) {
  clift.func @f<!f>() attributes {
    handle = "/function/0x40001001:Code_x86_64"
  } {
    %s = clift.local : !s$p
    %u = clift.local : !u$p

    // CHECK: var_0->x;
    clift.expr {
      %a = clift.access<indirect 0> %s : !s$p -> !int32_t
      clift.yield %a : !int32_t
    }

    // CHECK: var_1->x;
    clift.expr {
      %a = clift.access<indirect 0> %u : !u$p -> !int32_t
      clift.yield %a : !int32_t
    }

    // CHECK: var_0->y;
    clift.expr {
      %a = clift.access<indirect 1> %s : !s$p -> !int32_t
      clift.yield %a : !int32_t
    }

    // CHECK: var_1->y;
    clift.expr {
      %a = clift.access<indirect 1> %u : !u$p -> !int32_t
      clift.yield %a : !int32_t
    }

    // CHECK: (*var_0).x;
    clift.expr {
      %v = clift.indirection %s : !s$p
      %a = clift.access<0> %v : !s -> !int32_t
      clift.yield %a : !int32_t
    }

    // CHECK: (*var_1).x;
    clift.expr {
      %v = clift.indirection %u : !u$p
      %a = clift.access<0> %v : !u -> !int32_t
      clift.yield %a : !int32_t
    }
  }
  // CHECK: }
}
