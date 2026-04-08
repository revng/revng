//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s -emit-field-accesses -canonicalize 2>&1 | FileCheck %s

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

// Test the SizeRelation criterion: Same > Larger > Smaller

!u = !clift.union<
  "1" : {
    "" : !int64_t,
    "" : !int32_t,
    "" : !int16_t
  }
>

!struct_sizerelation = !clift.struct<
  "2" : size(16) {
    "" : offset(0) !u,
    "" : offset(8) !int64_t
  }
>

module attributes {clift.module} {
  // Access at offset 0 with size 8 - should select the int64_t field (same size)
  clift.func @test_sizerelation_same64<!f>() {
    %0 = clift.local : !struct_sizerelation
    clift.expr {
      %1 = clift.addressof %0 : !clift.ptr<8 to !struct_sizerelation>
      %2 = clift.bitcast %1 : !clift.ptr<8 to !struct_sizerelation> -> !clift.ptr<8 to !int64_t>
      clift.yield %2 : !clift.ptr<8 to !int64_t>
    }
  }

  // CHECK-LABEL: clift.func @test_sizerelation_same64<!f>
  // CHECK: [[STRUCT:%[0-9]+]] = clift.local : !_2_
  // CHECK: [[ADDR1:%[0-9]+]] = clift.addressof [[STRUCT]] : !clift.ptr<8 to !_2_>
  // CHECK: [[ACCESS1:%[0-9]+]] = clift.access<indirect 0> [[ADDR1]]
  // CHECK: [[ACCESS2:%[0-9]+]] = clift.access< 0> [[ACCESS1]]
  // CHECK: [[ADDR2:%[0-9]+]] = clift.addressof [[ACCESS2]]
  // CHECK: clift.yield [[ADDR2]] : !clift.ptr<8 to !int64_t>

  // Access at offset 0 with size 4 - should select the int32_t field (same size)
  clift.func @test_sizerelation_same32<!f>() {
    %0 = clift.local : !struct_sizerelation
    clift.expr {
      %1 = clift.addressof %0 : !clift.ptr<8 to !struct_sizerelation>
      %2 = clift.bitcast %1 : !clift.ptr<8 to !struct_sizerelation> -> !clift.ptr<8 to !int32_t>
      clift.yield %2 : !clift.ptr<8 to !int32_t>
    }
  }

  // CHECK-LABEL: clift.func @test_sizerelation_same32<!f>
  // CHECK: [[STRUCT:%[0-9]+]] = clift.local : !_2_
  // CHECK: [[ADDR1:%[0-9]+]] = clift.addressof [[STRUCT]] : !clift.ptr<8 to !_2_>
  // CHECK: [[ACCESS1:%[0-9]+]] = clift.access<indirect 0> [[ADDR1]]
  // CHECK: [[ACCESS2:%[0-9]+]] = clift.access< 1> [[ACCESS1]]
  // CHECK: [[ADDR2:%[0-9]+]] = clift.addressof [[ACCESS2]]
  // CHECK: clift.yield [[ADDR2]] : !clift.ptr<8 to !int32_t>

  // Access at offset 0 with size 2 should select the int16_t field (same size)
  clift.func @test_sizerelation_same16<!f>() {
    %0 = clift.local : !struct_sizerelation
    clift.expr {
      %1 = clift.addressof %0 : !clift.ptr<8 to !struct_sizerelation>
      %2 = clift.bitcast %1 : !clift.ptr<8 to !struct_sizerelation> -> !clift.ptr<8 to !int16_t>
      clift.yield %2 : !clift.ptr<8 to !int16_t>
    }
  }

  // CHECK-LABEL: clift.func @test_sizerelation_same16<!f>
  // CHECK: [[STRUCT:%[0-9]+]] = clift.local : !_2_
  // CHECK: [[ADDR1:%[0-9]+]] = clift.addressof [[STRUCT]] : !clift.ptr<8 to !_2_>
  // CHECK: [[ACCESS1:%[0-9]+]] = clift.access<indirect 0> [[ADDR1]]
  // CHECK: [[ACCESS2:%[0-9]+]] = clift.access< 2> [[ACCESS1]]
  // CHECK: [[ADDR2:%[0-9]+]] = clift.addressof [[ACCESS2]]
  // CHECK: clift.yield [[ADDR2]] : !clift.ptr<8 to !int16_t>
}
