//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s -emit-field-accesses -canonicalize 2>&1 | FileCheck %s

!void = !clift.void
!generic64_t = !clift.int<generic 8>
!int64_t = !clift.int<signed 8>
!int16_t = !clift.int<signed 2>
!int64_t$ptr = !clift.ptr<8 to !int64_t>
!int16_t$ptr = !clift.ptr<8 to !int16_t>

!f = !clift.func<
  "1000" as "f" : !void(!generic64_t)
>

// `struct` with array of `int64_t` at `offset_16`, accessed via `<< 3`
// (stride=8).
!s_stride8 = !clift.struct<
  "1" : size(96) {
    "" : offset(0) !generic64_t,
    "" : offset(8) !generic64_t,
    "" : offset(16) !clift.array<10 x !int64_t>
  }
>

// `struct` with an `array` of `int16_t` at `offset_8`, accessed via `<< 1`
// (stride=2).
!s_stride2 = !clift.struct<
  "2" : size(28) {
    "" : offset(0) !generic64_t,
    "" : offset(8) !clift.array<10 x !int16_t>
  }
>

module attributes {clift.module} {

  // Access pattern: `base + 16 + arg0 << 3` (stride=8 via shift-left by 3)
  // This exercises `composeShl` producing `Stride=8`, then `composeAdd` merging
  // it.
  clift.func @test_stride8_shl<!f>(%arg0 : !generic64_t) {
    %0 = clift.local : !s_stride8
    clift.expr {
      %1 = clift.addressof %0 : !clift.ptr<8 to !s_stride8>
      %2 = clift.bitcast %1 : !clift.ptr<8 to !s_stride8> -> !clift.ptr<8 to !void>
      %3 = clift.bitcast %2 : !clift.ptr<8 to !void> -> !generic64_t
      %4 = clift.imm 3 : !generic64_t
      %5 = clift.shl %arg0, %4 : !generic64_t
      %6 = clift.imm 16 : !generic64_t
      %7 = clift.add %5, %6 : !generic64_t
      %8 = clift.add %3, %7 : !generic64_t
      %9 = clift.bitcast %8 : !generic64_t -> !int64_t$ptr
      clift.yield %9 : !int64_t$ptr
    }
  }

  // CHECK-LABEL: clift.func @test_stride8_shl<!f>
  // CHECK-SAME: ([[ARG0:%[a-z0-9]+]]: !generic64_t)
  // CHECK: [[LOCAL:%[0-9]+]] = clift.local : !_1_
  // CHECK: [[ADDRESSOF1:%[0-9]+]] = clift.addressof [[LOCAL]] : !clift.ptr<8 to !_1_>
  // CHECK: [[ACCESS:%[0-9]+]] = clift.access<indirect 2> [[ADDRESSOF1]]
  // CHECK: [[CAST:%[0-9]+]] = clift.decay [[ACCESS]]
  // CHECK: [[SUBSCRIPT:%[0-9]+]] = clift.subscript [[CAST]], [[ARG0]]
  // CHECK: [[ADDRESSOF2:%[0-9]+]] = clift.addressof [[SUBSCRIPT]]
  // CHECK: clift.yield [[ADDRESSOF2]] : !clift.ptr<8 to !int64_t>

  // Access pattern: `base + 8 + arg0 << `1 (stride=2 via shift-left by 1)
  // This exercises `composeShl` producing `Stride=2`, then `composeAdd` merging
  // it.
  clift.func @test_stride2_shl<!f>(%arg0 : !generic64_t) {
    %0 = clift.local : !s_stride2
    clift.expr {
      %1 = clift.addressof %0 : !clift.ptr<8 to !s_stride2>
      %2 = clift.bitcast %1 : !clift.ptr<8 to !s_stride2> -> !clift.ptr<8 to !void>
      %3 = clift.bitcast %2 : !clift.ptr<8 to !void> -> !generic64_t
      %4 = clift.imm 1 : !generic64_t
      %5 = clift.shl %arg0, %4 : !generic64_t
      %6 = clift.imm 8 : !generic64_t
      %7 = clift.add %5, %6 : !generic64_t
      %8 = clift.add %3, %7 : !generic64_t
      %9 = clift.bitcast %8 : !generic64_t -> !int16_t$ptr
      clift.yield %9 : !int16_t$ptr
    }
  }

  // CHECK-LABEL: clift.func @test_stride2_shl<!f>
  // CHECK-SAME: ([[ARG0:%[a-z0-9]+]]: !generic64_t)
  // CHECK: [[LOCAL:%[0-9]+]] = clift.local : !_2_
  // CHECK: [[ADDRESSOF1:%[0-9]+]] = clift.addressof [[LOCAL]] : !clift.ptr<8 to !_2_>
  // CHECK: [[ACCESS:%[0-9]+]] = clift.access<indirect 1> [[ADDRESSOF1]]
  // CHECK: [[CAST:%[0-9]+]] = clift.decay [[ACCESS]]
  // CHECK: [[SUBSCRIPT:%[0-9]+]] = clift.subscript [[CAST]], [[ARG0]]
  // CHECK: [[ADDRESSOF2:%[0-9]+]] = clift.addressof [[SUBSCRIPT]]
  // CHECK: clift.yield [[ADDRESSOF2]] : !clift.ptr<8 to !int16_t>
}
