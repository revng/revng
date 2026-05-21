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

!union_typedistance = !clift.union<
  "1" : {
    "" : !int32_t,
    "" : !uint32_t,
    "" : !float32_t
  }
>

!struct_typedistance = !clift.struct<
  "2" : size(4) {
    "" : offset(0) !union_typedistance
  }
>

// The union now misses the `int32_t`, therefore we should revert to `uint32_t`
// as an alternative

!union_typedistance_2 = !clift.union<
  "3" : {
    "" : !uint32_t,
    "" : !float32_t
  }
>

!struct_typedistance_2 = !clift.struct<
  "4" : size(4) {
    "" : offset(0) !union_typedistance_2
  }
>

module attributes {clift.module} {
  // Access with signed type - should select int32_t field
  clift.func @test_typedistance1<!f>() {
    %0 = clift.local : !struct_typedistance
    clift.expr {
      %1 = clift.addressof %0 : !clift.ptr<8 to !struct_typedistance>
      %2 = clift.bitcast %1 : !clift.ptr<8 to !struct_typedistance> -> !clift.ptr<8 to !int32_t>
      clift.yield %2 : !clift.ptr<8 to !int32_t>
    }
  }

  // CHECK-LABEL: clift.func @test_typedistance1<!f>
  // CHECK: [[STRUCT:%[0-9]+]] = clift.local : !_2_
  // CHECK: [[ADDRESSOF1:%[0-9]+]] = clift.addressof [[STRUCT]] : !clift.ptr<8 to !_2_>
  // CHECK: [[ACCESS1:%[0-9]+]] = clift.access<indirect 0> [[ADDRESSOF1]]
  // CHECK: [[ACCESS2:%[0-9]+]] = clift.access< 0> [[ACCESS1]]
  // CHECK: [[ADDRESSOF2:%[0-9]+]] = clift.addressof [[ACCESS2]]
  // CHECK: clift.yield [[ADDRESSOF2]] : !clift.ptr<8 to !int32_t>

  // Access with unsigned type - should select uint32_t field
  clift.func @test_typedistance2<!f>() {
    %0 = clift.local : !struct_typedistance
    clift.expr {
      %1 = clift.addressof %0 : !clift.ptr<8 to !struct_typedistance>
      %2 = clift.bitcast %1 : !clift.ptr<8 to !struct_typedistance> -> !clift.ptr<8 to !uint32_t>
      clift.yield %2 : !clift.ptr<8 to !uint32_t>
    }
  }

  // CHECK-LABEL: clift.func @test_typedistance2<!f>
  // CHECK: [[STRUCT:%[0-9]+]] = clift.local : !_2_
  // CHECK: [[ADDRESSOF1:%[0-9]+]] = clift.addressof [[STRUCT]] : !clift.ptr<8 to !_2_>
  // CHECK: [[ACCESS1:%[0-9]+]] = clift.access<indirect 0> [[ADDRESSOF1]]
  // CHECK: [[ACCESS2:%[0-9]+]] = clift.access< 1> [[ACCESS1]]
  // CHECK: [[ADDRESSOF2:%[0-9]+]] = clift.addressof [[ACCESS2]]
  // CHECK: clift.yield [[ADDRESSOF2]] : !clift.ptr<8 to !uint32_t>

  // Access with float type - should select float32_t field
  clift.func @test_typedistance3<!f>() {
    %0 = clift.local : !struct_typedistance
    clift.expr {
      %1 = clift.addressof %0 : !clift.ptr<8 to !struct_typedistance>
      %2 = clift.bitcast %1 : !clift.ptr<8 to !struct_typedistance> -> !clift.ptr<8 to !float32_t>
      clift.yield %2 : !clift.ptr<8 to !float32_t>
    }
  }

  // CHECK-LABEL: clift.func @test_typedistance3<!f>
  // CHECK: [[STRUCT:%[0-9]+]] = clift.local : !_2_
  // CHECK: [[ADDRESSOF1:%[0-9]+]] = clift.addressof [[STRUCT]] : !clift.ptr<8 to !_2_>
  // CHECK: [[ACCESS1:%[0-9]+]] = clift.access<indirect 0> [[ADDRESSOF1]]
  // CHECK: [[ACCESS2:%[0-9]+]] = clift.access< 2> [[ACCESS1]]
  // CHECK: [[ADDRESSOF2:%[0-9]+]] = clift.addressof [[ACCESS2]]
  // CHECK: clift.yield [[ADDRESSOF2]] : !clift.ptr<8 to !float32_t>

  // Access with signed type - should select uint32_t field
  clift.func @test_typedistance4<!f>() {
    %0 = clift.local : !struct_typedistance
    clift.expr {
      %1 = clift.addressof %0 : !clift.ptr<8 to !struct_typedistance>
      %2 = clift.bitcast %1 : !clift.ptr<8 to !struct_typedistance> -> !clift.ptr<8 to !int32_t>
      clift.yield %2 : !clift.ptr<8 to !int32_t>
    }
  }

  // CHECK-LABEL: clift.func @test_typedistance4<!f>
  // CHECK: [[STRUCT:%[0-9]+]] = clift.local : !_2_
  // CHECK: [[ADDRESSOF1:%[0-9]+]] = clift.addressof [[STRUCT]] : !clift.ptr<8 to !_2_>
  // CHECK: [[ACCESS1:%[0-9]+]] = clift.access<indirect 0> [[ADDRESSOF1]]
  // CHECK: [[ACCESS2:%[0-9]+]] = clift.access< 0> [[ACCESS1]]
  // CHECK: [[ADDRESSOF2:%[0-9]+]] = clift.addressof [[ACCESS2]]
  // CHECK: clift.yield [[ADDRESSOF2]] : !clift.ptr<8 to !int32_t>
}
