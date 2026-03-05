//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s -emit-field-accesses -canonicalize 2>&1 | FileCheck %s

!void = !clift.void
!generic64_t = !clift.int<generic 8>
!int8_t = !clift.int<signed 1>
!int16_t = !clift.int<signed 2>
!int32_t = !clift.int<signed 4>
!uint32_t = !clift.int<unsigned 4>
!int8_t$ptr = !clift.ptr<8 to !int8_t>
!int16_t$ptr = !clift.ptr<8 to !int16_t>
!int32_t$ptr = !clift.ptr<8 to !int32_t>

// Generic void function prototype with no argument
!f = !clift.func<
  "1000" as "f" : !void()
>

!s = !clift.struct<
  "1" : size(8) {
    "" : offset(0) !int32_t,
    "" : offset(4) !int32_t
  }
>

// Struct access with a leftover offset

module attributes {clift.module} {
  clift.func @f<!f>() {
    %0 = clift.local : !s
    clift.expr {
      %1 = clift.addressof %0 : !clift.ptr<8 to !s>
      %2 = clift.bitcast %1 : !clift.ptr<8 to !s> -> !clift.ptr<8 to !void>
      %3 = clift.bitcast %2 : !clift.ptr<8 to !void> -> !generic64_t
      %4 = clift.imm 6 : !generic64_t
      %5 = clift.add %3, %4 : !generic64_t
      %6 = clift.bitcast %5 : !generic64_t -> !clift.ptr<8 to !int8_t>
      clift.yield %6 : !int8_t$ptr
    }
  }

  // CHECK-LABEL: clift.func @f<!f>
  // CHECK: [[STRUCT:%[0-9]+]] = clift.local : !_1_
  // CHECK: [[ADDRESSOF1:%[0-9]+]] = clift.addressof [[STRUCT]] : !clift.ptr<8 to !_1_>
  // CHECK: [[ACCESS:%[0-9]+]] = clift.access<indirect 1> [[ADDRESSOF1]]
  // CHECK: [[ADDRESSOF2:%[0-9]+]] = clift.addressof [[ACCESS]]
  // CHECK: [[CAST1:%[0-9]+]] = clift.bitcast [[ADDRESSOF2]]
  // CHECK: [[IMM:%[0-9]+]] = clift.imm 2
  // CHECK: [[ADD:%[0-9]+]] = clift.add [[CAST1]], [[IMM]]
  // CHECK: [[CAST2:%[0-9]+]] = clift.bitcast [[ADD]]
  // CHECK: [[CAST3:%[0-9]+]] = clift.bitcast [[CAST2]]
  // CHECK: clift.yield [[CAST3]] : !clift.ptr<8 to !int8_t>

  // Struct access going over the boundaries of the struct, not converted into an access

  clift.func @g<!f>() {
    %0 = clift.local : !s
    clift.expr {
      %1 = clift.addressof %0 : !clift.ptr<8 to !s>
      %2 = clift.bitcast %1 : !clift.ptr<8 to !s> -> !clift.ptr<8 to !void>
      %3 = clift.bitcast %2 : !clift.ptr<8 to !void> -> !generic64_t
      %4 = clift.imm 8 : !generic64_t
      %5 = clift.add %3, %4 : !generic64_t
      %6 = clift.bitcast %5 : !generic64_t -> !clift.ptr<8 to !int8_t>
      clift.yield %6 : !int8_t$ptr
    }
  }

  // CHECK-LABEL: clift.func @g<!f>
  // CHECK: [[STRUCT:%[0-9]+]] = clift.local : !_1_
  // CHECK: [[ADDRESSOF:%[0-9]+]] = clift.addressof [[STRUCT]] : !clift.ptr<8 to !_1_>
  // CHECK: [[CAST1:%[0-9]+]] = clift.bitcast [[ADDRESSOF]]
  // CHECK: [[CAST2:%[0-9]+]] = clift.bitcast [[CAST1]]
  // CHECK: [[IMM:%[0-9]+]] = clift.imm 8
  // CHECK: [[ADD:%[0-9]+]] = clift.add [[CAST2]], [[IMM]]
  // CHECK: [[CAST3:%[0-9]+]] = clift.bitcast [[ADD]]
  // CHECK: clift.yield [[CAST3]] : !clift.ptr<8 to !int8_t>
  // CHECK-NOT: [[ACCESS:%[0-9]+]] = clift.access<indirect 1> [[ADDRESSOF]]
}
