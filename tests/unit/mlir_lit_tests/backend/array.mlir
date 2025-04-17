//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s --emit-c="tagless model=%S/model.yml" -o /dev/null | FileCheck %s

!void = !clift.primitive<void 0>

!int32_t = !clift.primitive<signed 4>
!int32_t$p = !clift.ptr<8 to !int32_t>

!int32_t$1 = !clift.array<1 x !int32_t>
!int32_t$1$p = !clift.ptr<8 to !int32_t$1>

!f = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" : !void()
>

module attributes {clift.module} {
  // CHECK: void fun_0x40001001(void) {
  clift.func @f<!f>() attributes {
    handle = "/function/0x40001001:Code_x86_64"
  } {
    // CHECK: int32_t _var_0[1];
    %array = clift.local !int32_t$1 "array"

    // CHECK: _var_0[0];
    clift.expr {
      %p = clift.cast<decay> %array : !int32_t$1 -> !int32_t$p
      %i = clift.imm 0 : !int32_t
      %r = clift.subscript %p, %i : (!int32_t$p, !int32_t)
      clift.yield %r : !int32_t
    }

    // CHECK: int32_t(*_var_1)[1]
    %p_array = clift.local !int32_t$1$p "p_array" = {
      %r = clift.addressof %array : !int32_t$1$p
      clift.yield %r : !int32_t$1$p
    }

    // CHECK: (*_var_1)[(0, 0)]
    clift.expr {
      %q = clift.indirection %p_array : !int32_t$1$p
      %p = clift.cast<decay> %q : !int32_t$1 -> !int32_t$p
      %i = clift.imm 0 : !int32_t
      %comma = clift.comma %i, %i : !int32_t, !int32_t
      %r = clift.subscript %p, %comma : (!int32_t$p, !int32_t)
      clift.yield %r : !int32_t
    }
  }
  // CHECK: }
}
