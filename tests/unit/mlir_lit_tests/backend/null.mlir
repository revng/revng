//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt --emit-c %s | FileCheck %s
// RUN: %revngcliftopt --emit-c=ptml %s -o /dev/null | %revngptml | FileCheck %s

!void = !clift.void
!int64_t = !clift.int<signed 8>
!int64_t$ptr = !clift.ptr<8 to !int64_t>

!f = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" : !void()
>

module attributes {clift.module} {
  // CHECK: void fun_0x40001001(void) {
  clift.func @fun_0x40001001<!f>() attributes {
    handle = "/function/0x40001001:Code_x86_64"
  } {
    // CHECK: (int64_t *) NULL;
    clift.expr {
        %0 = clift.imm 0 : !int64_t
        %1 = clift.bitcast %0 : !int64_t -> !int64_t$ptr
        clift.yield %1 : !int64_t$ptr
    }
  }
  // CHECK: }
}
