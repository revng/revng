//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s --emit-c="tagless model=%S/model.yml" -o /dev/null | FileCheck %s

!void = !clift.primitive<void 0>
!int32_t = !clift.primitive<signed 4>

!f = !clift.defined<#clift.func<"/model-type/1001" : !void()>>

clift.module {
  // CHECK: void fun_0x40001001(void) {
  clift.func @f<!f>() attributes {
    handle = "/function/0x40001001:Code_x86_64"
  } {
    // CHECK: for (;;)
    clift.for {} {} {} {
      // CHECK: 1;
      clift.expr {
        %1 = clift.imm 1 : !int32_t
        clift.yield %1 : !int32_t
      }
    }

    // CHECK: for (; 2; 3) {
    clift.for {} {
      %2 = clift.imm 2 : !int32_t
      clift.yield %2 : !int32_t
    }{
      %3 = clift.imm 3 : !int32_t
      clift.yield %3 : !int32_t
    } {
      // CHECK: 4;
      clift.expr {
        %4 = clift.imm 4 : !int32_t
        clift.yield %4 : !int32_t
      }
      // CHECK: 5;
      clift.expr {
        %5 = clift.imm 5 : !int32_t
        clift.yield %5 : !int32_t
      }
    }
    // CHECK: }
  }
  // CHECK: }
}
