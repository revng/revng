//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt --emit-c %s | FileCheck %s
// RUN: %revngcliftopt --emit-c=ptml %s -o /dev/null | %revngptml | FileCheck %s

!void = !clift.primitive<void 0>

!int64_t = !clift.primitive<signed 8>
!uint64_t = !clift.primitive<unsigned 8>

!uint64_t$ptr = !clift.ptr<8 to !uint64_t>

!f = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" : !void()
>

module attributes {clift.module} {
  // CHECK: void fun_0x40001001(void) {
  clift.func @fun_0x40001001<!f>() attributes {
    handle = "/function/0x40001001:Code_x86_64"
  } {
    %x = clift.local : !uint64_t attributes {
      handle = "/local-variable/0x40001001:Code_x86_64/0",
      clift.name = "var_0"
    }

    // CHECK: (uint64_t)0L;
    clift.expr {
      %0 = clift.imm 0 : !int64_t
      %1 = clift.cast<bitcast> %0 : !int64_t -> !uint64_t
      clift.yield %1 : !uint64_t
    }

    // CHECK: (uint64_t)&var_0;
    clift.expr {
      %0 = clift.addressof %x : !uint64_t$ptr
      %1 = clift.cast<bitcast> %0 : !uint64_t$ptr -> !uint64_t
      clift.yield %1 : !uint64_t
    }
  }
  // CHECK: }
}
