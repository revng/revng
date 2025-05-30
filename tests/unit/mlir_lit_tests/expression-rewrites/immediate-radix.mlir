//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s --deduce-immediate-radices | FileCheck %s

!void = !clift.primitive<void 0>
!int32_t = !clift.primitive<signed 4>

!f = !clift.func<"/model-type/1001" : !void()>

module attributes {clift.module} {
  clift.func @f<!f>() -> !void {
    // 0
    // CHECK: clift.expr {
    clift.expr {
      // CHECK: %0 = clift.imm 0 : !int32_t
      %0 = clift.imm 0 : !int32_t
      // CHECK: clift.yield %0 : !int32_t
      clift.yield %0 : !int32_t
    }
    // CHECK: }

    // 0x100
    // CHECK: clift.expr {
    clift.expr {
      // CHECK: %0 = clift.imm 256 {clift.radix = 16 : ui32} : !int32_t
      %0 = clift.imm 256 : !int32_t
      // CHECK: clift.yield %0 : !int32_t
      clift.yield %0 : !int32_t
    }
    // CHECK: }

    // 1000
    // CHECK: clift.expr {
    clift.expr {
      // CHECK: %0 = clift.imm 1000 : !int32_t
      %0 = clift.imm 1000 : !int32_t
      // CHECK: clift.yield %0 : !int32_t
      clift.yield %0 : !int32_t
    }
    // CHECK: }

    // 0xCCCC
    // CHECK: clift.expr {
    clift.expr {
      // CHECK: %0 = clift.imm 52428 {clift.radix = 16 : ui32} : !int32_t
      %0 = clift.imm 52428 : !int32_t
      // CHECK: clift.yield %0 : !int32_t
      clift.yield %0 : !int32_t
    }
    // CHECK: }
  }
}
