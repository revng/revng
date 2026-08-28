//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s -emit-field-accesses -canonicalize 2>&1 | FileCheck %s

!void = !clift.void
!generic64_t = !clift.int<generic 8>
!int32_t = !clift.int<signed 4>
!int32_t$ptr = !clift.ptr<8 to !int32_t>

!f = !clift.func<
  "1000" as "f" : !void(!generic64_t, !generic64_t)
>

// A nested 2D array: `array<10 x array<5 x int32_t>>`. The outer array has
// stride 20 (5 * `sizeof(int32_t)`), the inner array has stride 4.
!a = !clift.array<10 x !clift.array<5 x !int32_t>>

!a2 = !clift.array<5 x !int32_t>
!s = !clift.struct<
  "1" : size(20) {
    "" : offset(0) !a2
  }
>
!a3 = !clift.array<10 x !s>

module attributes {clift.module} {

  // Access pattern: `base + i * 20 + j * 4`.
  // Two-term LinearCombination representing `array[i][j]`.
  clift.func @test_nested_array<!f>(%arg0 : !generic64_t, %arg1 : !generic64_t) {
    %0 = clift.local : !a
    clift.expr {
      %1 = clift.addressof %0 : !clift.ptr<8 to !a>
      %2 = clift.bitcast %1 : !clift.ptr<8 to !a> -> !clift.ptr<8 to !void>
      %3 = clift.bitcast %2 : !clift.ptr<8 to !void> -> !generic64_t
      // Outer index: `i * 20`
      %4 = clift.imm 20 : !generic64_t
      %5 = clift.mul %arg0, %4 : !generic64_t
      // Inner index: `j * 4`
      %6 = clift.imm 4 : !generic64_t
      %7 = clift.mul %arg1, %6 : !generic64_t
      // Combined: `base + i * 20 + j * 4`
      %8 = clift.add %5, %7 : !generic64_t
      %9 = clift.add %3, %8 : !generic64_t
      %10 = clift.bitcast %9 : !generic64_t -> !int32_t$ptr
      clift.yield %10 : !int32_t$ptr
    }
  }

  // CHECK-LABEL: clift.func @test_nested_array<!f>
  // CHECK-SAME: ([[ARG0:%[a-z0-9]+]]: !generic64_t, [[ARG1:%[a-z0-9]+]]: !generic64_t)
  // CHECK: [[LOCAL:%[0-9]+]] = clift.local : !clift.array<10 x !clift.array<5 x !int32_t>>
  // CHECK: [[ADDRESSOF1:%[0-9]+]] = clift.addressof [[LOCAL]] : !clift.ptr<8 to !clift.array<10 x !clift.array<5 x !int32_t>>>
  // CHECK: [[INDIRECTION:%[0-9]+]] = clift.indirection [[ADDRESSOF1]]
  // CHECK: [[DECAY1:%[0-9]+]] = clift.decay [[INDIRECTION]]
  // CHECK: [[SUBSCRIPTION1:%[0-9]+]] = clift.subscript [[DECAY1]], [[ARG0]]
  // CHECK: [[DECAY2:%[0-9]+]] = clift.decay [[SUBSCRIPTION1]]
  // CHECK: [[SUBSCRIPTION2:%[0-9]+]] = clift.subscript [[DECAY2]], [[ARG1]]
  // CHECK: [[ADDRESSOF2:%[0-9]+]] = clift.addressof [[SUBSCRIPTION2]]
  // CHECK: clift.yield [[ADDRESSOF2]] : !clift.ptr<8 to !int32_t>

  // Same access pattern but with a `struct` wrapping the `inner array`.
  // This exercises the path through struct access + array subscript at each
  // nesting level.
  clift.func @test_nested_struct_with_array<!f>(%arg0 : !generic64_t, %arg1 : !generic64_t) {
    %0 = clift.local : !a3
    clift.expr {
      %1 = clift.addressof %0 : !clift.ptr<8 to !a3>
      %2 = clift.bitcast %1 : !clift.ptr<8 to !a3> -> !clift.ptr<8 to !void>
      %3 = clift.bitcast %2 : !clift.ptr<8 to !void> -> !generic64_t
      // Outer index: `i * 20`
      %4 = clift.imm 20 : !generic64_t
      %5 = clift.mul %arg0, %4 : !generic64_t
      // Inner index: `j * 4`
      %6 = clift.imm 4 : !generic64_t
      %7 = clift.mul %arg1, %6 : !generic64_t
      // Combined: `base + i * 20 + j * 4`
      %8 = clift.add %5, %7 : !generic64_t
      %9 = clift.add %3, %8 : !generic64_t
      %10 = clift.bitcast %9 : !generic64_t -> !int32_t$ptr
      clift.yield %10 : !int32_t$ptr
    }
  }

  // CHECK-LABEL: clift.func @test_nested_struct_with_array<!f>
  // CHECK-SAME: ([[ARG0:%[a-z0-9]+]]: !generic64_t, [[ARG1:%[a-z0-9]+]]: !generic64_t)
  // CHECK: [[LOCAL:%[0-9]+]] = clift.local
  // CHECK: [[ADDRESSOF1:%[0-9]+]] = clift.addressof [[LOCAL]]
  // CHECK: [[INDIRECTION:%[0-9]+]] = clift.indirection [[ADDRESSOF1]]
  // CHECK: [[DECAY1:%[0-9]+]] = clift.decay [[INDIRECTION]]
  // CHECK: [[SUBSCRIPTION1:%[0-9]+]] = clift.subscript [[DECAY1]], [[ARG0]]
  // CHECK: [[ACCESS:%[0-9]+]] = clift.access<0> [[SUBSCRIPTION1]]
  // CHECK: [[DECAY2:%[0-9]+]] = clift.decay [[ACCESS]]
  // CHECK: [[SUBSCRIPTION2:%[0-9]+]] = clift.subscript [[DECAY2]], [[ARG1]]
  // CHECK: [[ADDRESSOF2:%[0-9]+]] = clift.addressof [[SUBSCRIPTION2]]
  // CHECK: clift.yield [[ADDRESSOF2]] : !clift.ptr<8 to !int32_t>
}
