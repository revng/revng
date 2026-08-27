//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s -emit-field-accesses -canonicalize 2>&1 | FileCheck %s

!void = !clift.void
!generic64_t = !clift.int<generic 8>
!int8_t = !clift.int<signed 1>
!int16_t = !clift.int<signed 2>
!int32_t = !clift.int<signed 4>
!int64_t = !clift.int<signed 8>
!uint8_t = !clift.int<unsigned 1>
!uint16_t = !clift.int<unsigned 2>
!uint32_t = !clift.int<unsigned 4>
!uint64_t = !clift.int<unsigned 8>
!float32_t = !clift.float<4>
!float64_t = !clift.float<8>

// Generic void function prototype with no argument
!f = !clift.func<
  "1000" as "f" : !void()
>

!struct_startdistance = !clift.struct<
  "1" : size(16) {
    "" : offset(0) !int32_t,
    "" : offset(4) !int64_t,
    "" : offset(12) !int32_t
  }
>

// Access at offset 6 in the `struct`, between fields at offset 4 and 12, so we
// need to choose field at offset 4, plus a  leftover. Choosing field at offset
// 12 would lead to a negative `StartDistance`, which should result in an
// invalid score and not be selected

module attributes {clift.module} {
  // Access, of size 4, at offset 6 - should select field at offset 4 (int64_t),
  // because the end distance is within the boundaries of field
  clift.func @test_startdistance<!f>() {
    %0 = clift.local : !struct_startdistance
    clift.expr {
      %1 = clift.addressof %0 : !clift.ptr<8 to !struct_startdistance>
      %2 = clift.bitcast %1 : !clift.ptr<8 to !struct_startdistance> -> !generic64_t
      %3 = clift.imm 6 : !generic64_t
      %4 = clift.add %2, %3 : !generic64_t
      %5 = clift.bitcast %4 : !generic64_t -> !clift.ptr<8 to !int32_t>
      clift.yield %5 : !clift.ptr<8 to !int32_t>
    }
  }

  // CHECK-LABEL: clift.func @test_startdistance<!f>
  // CHECK: [[STRUCT:%[0-9]+]] = clift.local : !_1_
  // CHECK: [[ADDRESSOF1:%[0-9]+]] = clift.addressof [[STRUCT]] : !clift.ptr<8 to !_1_>
  // CHECK: [[ACCESS1:%[0-9]+]] = clift.ptr_access<1> [[ADDRESSOF1]]
  // CHECK: [[ADDRESSOF2:%[0-9]+]] = clift.addressof [[ACCESS1]]
  // CHECK: [[CAST1:%[0-9]+]] = clift.bitcast [[ADDRESSOF2]]
  // CHECK: [[IMM:%[0-9]+]] = clift.imm 2
  // CHECK: [[ADD:%[0-9]+]] = clift.add [[CAST1]], [[IMM]]
  // CHECK: [[CAST2:%[0-9]+]] = clift.bitcast [[ADD]]
  // CHECK: [[CAST3:%[0-9]+]] = clift.bitcast [[CAST2]]
  // CHECK: clift.yield [[CAST3]] : !clift.ptr<8 to !int32_t>

  // Access, of size 8, at offset 6 - should not select field at offset 4
  // (int64_t), because the end distance would overshoot the field, since they
  // only partially overlap
  clift.func @test_enddistance<!f>() {
    %0 = clift.local : !struct_startdistance
    clift.expr {
      %1 = clift.addressof %0 : !clift.ptr<8 to !struct_startdistance>
      %2 = clift.bitcast %1 : !clift.ptr<8 to !struct_startdistance> -> !generic64_t
      %3 = clift.imm 6 : !generic64_t
      %4 = clift.add %2, %3 : !generic64_t
      %5 = clift.bitcast %4 : !generic64_t -> !clift.ptr<8 to !int64_t>
      clift.yield %5 : !clift.ptr<8 to !int64_t>
    }
  }

  // CHECK-LABEL: clift.func @test_enddistance<!f>
  // CHECK: [[STRUCT:%[0-9]+]] = clift.local : !_1_
  // CHECK: [[ADDRESSOF1:%[0-9]+]] = clift.addressof [[STRUCT]] : !clift.ptr<8 to !_1_>
  // CHECK: [[CAST1:%[0-9]+]] = clift.bitcast [[ADDRESSOF1]]
  // CHECK: [[IMM:%[0-9]+]] = clift.imm 6
  // CHECK: [[ADD:%[0-9]+]] = clift.add [[CAST1]], [[IMM]]
  // CHECK: [[CAST2:%[0-9]+]] = clift.bitcast [[ADD]]
  // CHECK: clift.yield [[CAST2]] : !clift.ptr<8 to !int64_t>
  // CHECK-NOT: [[ACCESS1:%[0-9]+]] = clift.ptr_access<1> [[ADDRESSOF1]]
}
