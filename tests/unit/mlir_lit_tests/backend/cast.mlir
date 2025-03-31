//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s --emit-c="tagless model=%S/model.yml" -o /dev/null | FileCheck %s

!void = !clift.primitive<void 0>

!int32_t = !clift.primitive<signed 4>
!uint32_t = !clift.primitive<unsigned 4>

!uint32_t$ptr = !clift.ptr<4 to !uint32_t>

!f = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" : !void()
>

clift.module {
  // CHECK: void fun_0x40001001(void) {
  clift.func @f<!f>() attributes {
    handle = "/function/0x40001001:Code_x86_64"
  } {
    %x = clift.local !uint32_t "x"

    // CHECK: (uint32_t)0;
    clift.expr {
      %0 = clift.imm 0 : !int32_t
      %1 = clift.cast<bitcast> %0 : !int32_t -> !uint32_t
      clift.yield %1 : !uint32_t
    }

    // CHECK: (uint32_t)&_var_0;
    clift.expr {
      %0 = clift.addressof %x : !uint32_t$ptr
      %1 = clift.cast<bitcast> %0 : !uint32_t$ptr -> !uint32_t
      clift.yield %1 : !uint32_t
    }
  }
  // CHECK: }
}
