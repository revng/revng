//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s -emit-field-accesses -canonicalize 2>&1 | FileCheck %s

!void = !clift.void
!generic64_t = !clift.int<generic 8>
!int32_t = !clift.int<signed 4>
!int32_t$ptr = !clift.ptr<8 to !int32_t>

// Generic void function prototype with no argument
!f = !clift.func<
  "1000" as "f" : !void()
>

// Nested array where both levels have the same stride (4 bytes):
// int array[1][1]
// The inner array has stride 4 (1 element * 4 bytes), the outer array also has
// stride 4 (1 * 4 bytes). Both ArrayShapes have identical Stride=4.

// This tests that the `std::multiset` preserves duplicate `ArrayShapes` (a
// `std::set` would collapse them into one).
!inner = !clift.array<1 x !int32_t>
!outer = !clift.array<1 x !inner>

// Wrap in a struct so we have a clear base pointer context
!s = !clift.struct<
  "1" : size(8) {
    "" : offset(0) !int32_t,
    "" : offset(4) !outer
  }
>

module attributes {clift.module} {
  // Access to struct.array[1][1] at offset 4.
  // Should produce a struct access followed by two nested subscript operations.
  clift.func @test_nested_same_stride<!f>() {
    %0 = clift.local : !s
    clift.expr {
      %1 = clift.addressof %0 : !clift.ptr<8 to !s>
      %2 = clift.bitcast %1 : !clift.ptr<8 to !s> -> !clift.ptr<8 to !void>
      %3 = clift.bitcast %2 : !clift.ptr<8 to !void> -> !generic64_t
      %4 = clift.imm 4 : !generic64_t
      %5 = clift.add %3, %4 : !generic64_t
      %6 = clift.bitcast %5 : !generic64_t -> !clift.ptr<8 to !int32_t>
      clift.yield %6 : !int32_t$ptr
    }
  }

  // CHECK-LABEL: clift.func @test_nested_same_stride<!f>
  // CHECK: [[STRUCT:%[0-9]+]] = clift.local : !_1_
  // CHECK: [[ADDRESSOF1:%[0-9]+]] = clift.addressof [[STRUCT]] : !clift.ptr<8 to !_1_>
  // CHECK: [[ACCESS:%[0-9]+]] = clift.access<indirect 1> [[ADDRESSOF1]]
  // CHECK: [[CAST1:%[0-9]+]] = clift.decay [[ACCESS]]
  // CHECK: [[IMM1:%[0-9]+]] = clift.imm 0
  // CHECK: [[SUBSCRIPT1:%[0-9]+]] = clift.subscript [[CAST1]], [[IMM1]]
  // CHECK: [[CAST2:%[0-9]+]] = clift.decay [[SUBSCRIPT1]]
  // CHECK: [[IMM2:%[0-9]+]] = clift.imm 0
  // CHECK: [[SUBSCRIPT2:%[0-9]+]] = clift.subscript [[CAST2]], [[IMM2]]
  // CHECK: [[ADDRESSOF2:%[0-9]+]] = clift.addressof [[SUBSCRIPT2]]
  // CHECK: clift.yield [[ADDRESSOF2]] : !clift.ptr<8 to !int32_t>
}
