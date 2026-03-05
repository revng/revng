//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s --deduce-immediate-radices | FileCheck %s

!void = !clift.void
!int32_t = !clift.int<signed 4>

!f = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" : !void()
>

module attributes {clift.module} {
  clift.func @f<!f>() {
    // CHECK: clift.expr {
    clift.expr {
      // COM: 123456789 = 0x75BCD15
      // CHECK: %0 = clift.imm 123456789 : !int32_t
      %0 = clift.imm 123456789 : !int32_t
      // CHECK: clift.yield %0 : !int32_t
      clift.yield %0 : !int32_t
    // CHECK: }
    }

    // CHECK: clift.expr {
    clift.expr {
      // COM: 11185083 = 0xAAABBB
      // CHECK: %0 = clift.imm 11185083 {clift.radix = 16 : ui32} : !int32_t
      %0 = clift.imm 11185083 : !int32_t
      // CHECK: clift.yield %0 : !int32_t
      clift.yield %0 : !int32_t
    // CHECK: }
    }

    // CHECK: clift.expr {
    clift.expr {
      // CHECK: %0 = clift.imm 1 {clift.radix = 16 : ui32} : !int32_t
      %0 = clift.imm 1 : !int32_t
      // CHECK: %1 = clift.imm 1000 {clift.radix = 16 : ui32} : !int32_t
      %1 = clift.imm 1000 : !int32_t
      // CHECK: %2 = clift.bitor %0, %1 : !int32_t
      %2 = clift.bitor %0, %1 : !int32_t
      // CHECK: clift.yield %2 : !int32_t
      clift.yield %2 : !int32_t
    // CHECK: }
    }
  }
}
