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

!inner_struct = !clift.struct<
  "1" : size(4) {
    "" : offset(0) !int32_t
  }
>

!union_depth = !clift.union<
  "2" : {
    "" : !inner_struct,
    "" : !int32_t
  }
>

!struct_depth = !clift.struct<
  "3" : size(4) {
    "" : offset(0) !union_depth
  }
>

module attributes {clift.module} {
  // Access at offset 0 with int32_t, which should select the shallow union field
  // (depth 2: struct -> union -> int32_t) over the deep path
  // (depth 3: struct -> union -> inner_struct -> int32_t)
  clift.func @test_depth<!f>() {
    %0 = clift.local : !struct_depth
    clift.expr {
      %1 = clift.addressof %0 : !clift.ptr<8 to !struct_depth>
      %2 = clift.bitcast %1 : !clift.ptr<8 to !struct_depth> -> !clift.ptr<8 to !int32_t>
      clift.yield %2 : !clift.ptr<8 to !int32_t>
    }
  }

  // CHECK-LABEL: clift.func @test_depth<!f>
  // CHECK: [[STRUCT:%[0-9]+]] = clift.local : !_3_
  // CHECK: [[ADDRESSOF1:%[0-9]+]] = clift.addressof [[STRUCT]] : !clift.ptr<8 to !_3_>
  // CHECK: [[ACCESS1:%[0-9]+]] = clift.access<indirect 0> [[ADDRESSOF1]]
  // CHECK: [[ACCESS2:%[0-9]+]] = clift.access< 1> [[ACCESS1]]
  // CHECK: [[ADDRESSOF2:%[0-9]+]] = clift.addressof [[ACCESS2]]
  // CHECK: clift.yield [[ADDRESSOF2]] : !clift.ptr<8 to !int32_t>
}
