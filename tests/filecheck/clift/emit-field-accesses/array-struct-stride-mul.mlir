//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s -emit-field-accesses -canonicalize 2>&1 | FileCheck %s

!void = !clift.void
!generic64_t = !clift.int<generic 8>
!int32_t = !clift.int<signed 4>
!int32_t$ptr = !clift.ptr<8 to !int32_t>

!f = !clift.func<
  "1000" as "f" : !void(!generic64_t)
>

// A 20-byte inner `struct`, to be used as `array` element
!inner = !clift.struct<
  "2" : size(20) {
    "" : offset(0) !int32_t,
    "" : offset(4) !int32_t,
    "" : offset(8) !int32_t,
    "" : offset(12) !int32_t,
    "" : offset(16) !int32_t
  }
>

// Outer `struct` containing some fields before the `array`, an `array` of 10
// inner `struct`s with stride 20, and another field.
!outer = !clift.struct<
  "1" : size(220) {
    "" : offset(0) !generic64_t,
    "" : offset(8) !generic64_t,
    "" : offset(16) !clift.array<10 x !inner>,
    "" : offset(216) !int32_t
  }
>

module attributes {clift.module} {

  // Access pattern: `base + 16 + arg0 * 20`
  clift.func @test_stride20_mul<!f>(%arg0 : !generic64_t) {
    %0 = clift.local : !outer
    clift.expr {
      %1 = clift.addressof %0 : !clift.ptr<8 to !outer>
      %2 = clift.bitcast %1 : !clift.ptr<8 to !outer> -> !clift.ptr<8 to !void>
      %3 = clift.bitcast %2 : !clift.ptr<8 to !void> -> !generic64_t
      %4 = clift.imm 20 : !generic64_t
      %5 = clift.mul %4, %arg0 : !generic64_t
      %6 = clift.imm 16 : !generic64_t
      %7 = clift.add %5, %6 : !generic64_t
      %8 = clift.add %3, %7 : !generic64_t
      %9 = clift.bitcast %8 : !generic64_t -> !int32_t$ptr
      clift.yield %9 : !int32_t$ptr
    }
  }

  // CHECK-LABEL: clift.func @test_stride20_mul<!f>
  // CHECK-SAME: ([[ARG0:%[a-z0-9]+]]: !generic64_t)
  // CHECK: [[LOCAL:%[0-9]+]] = clift.local : !_1_
  // CHECK: [[ADDRESSOF1:%[0-9]+]] = clift.addressof [[LOCAL]] : !clift.ptr<8 to !_1_>
  // CHECK: [[ACCESS1:%[0-9]+]] = clift.ptr_access<2> [[ADDRESSOF1]]
  // CHECK: [[DECAY:%[0-9]+]] = clift.decay [[ACCESS1]]
  // CHECK: [[SUBSCRIPT:%[0-9]+]] = clift.subscript [[DECAY]], [[ARG0]]
  // CHECK: [[ACCESS2:%[0-9]+]] = clift.access<0> [[SUBSCRIPT]]
  // CHECK: [[ADDRESSOF2:%[0-9]+]] = clift.addressof [[ACCESS2]]
  // CHECK: clift.yield [[ADDRESSOF2]] : !clift.ptr<8 to !int32_t>
}
