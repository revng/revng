//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt --emit-c %s | FileCheck %s
// RUN: %revngcliftopt --emit-c=ptml %s -o /dev/null | %revngptml | FileCheck %s

!void = !clift.void

!int32_t = !clift.int<signed 4>
!uint32_t = !clift.int<unsigned 4>
!float32_t = !clift.float<4>

!uint32_t$ptr = !clift.ptr<4 to !uint32_t>

!f = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" : !void()
>

module attributes {clift.module} {
  // CHECK: void fun_0x40001001(void) {
  clift.func @fun_0x40001001<!f>() attributes {
    handle = "/function/0x40001001:Code_x86_64"
  } {
    %x = clift.local : !uint32_t attributes {
      handle = "/local-variable/0x40001001:Code_x86_64/0",
      name = "var_0"
    }

    // CHECK: (uint32_t) 0;
    clift.expr {
      %0 = clift.imm 0 : !int32_t
      %1 = clift.bitcast %0 : !int32_t -> !uint32_t
      clift.yield %1 : !uint32_t
    }

    // CHECK: (uint32_t) &var_0;
    clift.expr {
      %0 = clift.addressof %x : !uint32_t$ptr
      %1 = clift.bitcast %0 : !uint32_t$ptr -> !uint32_t
      clift.yield %1 : !uint32_t
    }

    // CHECK: bit_cast(float32_t, 0)
    clift.expr {
      %0 = clift.imm 0 : !int32_t
      %1 = clift.bitcast %0 : !int32_t -> !float32_t
      clift.yield %1 : !float32_t
    }
  }
  // CHECK: }
}
