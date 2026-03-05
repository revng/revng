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
// int array[3][1]
// The outer array has stride 4 (1 * sizeof(int)), the inner array also has
// stride 4 (sizeof(int)). Both ArrayShapes have Stride=4.
// This exercises `commonPrefixStrides` counting duplicate strides correctly:
// both levels must be matched for the score to reflect the full traversal
// depth.
!inner = !clift.array<1 x !int32_t>
!outer = !clift.array<3 x !inner>

module attributes {clift.module} {
  // Constant access: base + 4 represents array[1][0]
  clift.func @test_common_strides_duplicates<!f>() {
    %0 = clift.local : !outer
    clift.expr {
      %1 = clift.addressof %0 : !clift.ptr<8 to !outer>
      %2 = clift.bitcast %1 : !clift.ptr<8 to !outer> -> !clift.ptr<8 to !void>
      %3 = clift.bitcast %2 : !clift.ptr<8 to !void> -> !generic64_t
      %4 = clift.imm 4 : !generic64_t
      %5 = clift.add %3, %4 : !generic64_t
      %6 = clift.bitcast %5 : !generic64_t -> !int32_t$ptr
      clift.yield %6 : !int32_t$ptr
    }
  }

  // Both nested array levels should be traversed: outer[1][0]
  // CHECK-LABEL: clift.func @test_common_strides_duplicates<!f>
  // CHECK: [[ARRAY:%[0-9]+]] = clift.local : !clift.array<3 x !clift.array<1 x !int32_t>>
  // CHECK: [[ADDRESSOF1:%[0-9]+]] = clift.addressof [[ARRAY]] : !clift.ptr<8 to !clift.array<3 x !clift.array<1 x !int32_t>>>
  // CHECK: [[INDIR:%[0-9]+]] = clift.indirection [[ADDRESSOF1]]
  // CHECK: [[CAST1:%[0-9]+]] = clift.decay [[INDIR]]
  // CHECK: [[IMM1:%[0-9]+]] = clift.imm 1
  // CHECK: [[SUBSCRIPT1:%[0-9]+]] = clift.subscript [[CAST1]], [[IMM1]]
  // CHECK: [[CAST2:%[0-9]+]] = clift.decay [[SUBSCRIPT1]]
  // CHECK: [[IMM2:%[0-9]+]] = clift.imm 0
  // CHECK: [[SUBSCRIPT2:%[0-9]+]] = clift.subscript [[CAST2]], [[IMM2]]
  // CHECK: [[ADDRESSOF2:%[0-9]+]] = clift.addressof [[SUBSCRIPT2]]
  // CHECK: clift.yield [[ADDRESSOF2]] : !clift.ptr<8 to !int32_t>
}
