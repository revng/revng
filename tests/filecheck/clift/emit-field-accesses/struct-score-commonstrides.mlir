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

// Create nested arrays with different stride patterns
// One path: array[10] of array[5] of int32_t (strides: 20, 4)
// Another path: array[50] of int32_t (stride: 4)
// Access with strides [20, 4] should prefer the first path (2 common strides
// vs 1 stride)

!array_5_int32 = !clift.array<5 x !int32_t>
!array_10_of_array_5 = !clift.array<10 x !array_5_int32>
!array_50_int32 = !clift.array<50 x !int32_t>

!union_commonstrides = !clift.union<
  "1" : {
    "" : !array_50_int32,
    "" : !array_10_of_array_5
  }
>

!struct_commonstrides = !clift.struct<
  "2" : size(200) {
    "" : offset(0) !union_commonstrides
  }
>

module attributes {clift.module} {
  // Access at offset 28 (which is 20 + 8 = array[1][2])
  // Should select the nested array path due to more strides matching
  clift.func @test_commonstrides<!f>() {
    %0 = clift.local : !struct_commonstrides
    clift.expr {
      %1 = clift.addressof %0 : !clift.ptr<8 to !struct_commonstrides>
      %2 = clift.bitcast %1 : !clift.ptr<8 to !struct_commonstrides> -> !generic64_t
      %3 = clift.imm 28 : !generic64_t
      %4 = clift.add %2, %3 : !generic64_t
      %5 = clift.bitcast %4 : !generic64_t -> !clift.ptr<8 to !int32_t>
      clift.yield %5 : !clift.ptr<8 to !int32_t>
    }
  }

  // CHECK-LABEL: clift.func @test_commonstrides<!f>
  // CHECK: [[STRUCT:%[0-9]+]] = clift.local : !_2_
  // CHECK: [[ADDRESSOF1:%[0-9]+]] = clift.addressof [[STRUCT]] : !clift.ptr<8 to !_2_>
  // CHECK: [[ACCESS1:%[0-9]+]] = clift.access<indirect 0> [[ADDRESSOF1]]
  // CHECK: [[ACCESS2:%[0-9]+]] = clift.access< 1> [[ACCESS1]]
  // CHECK: [[CAST1:%[0-9]+]] = clift.decay [[ACCESS2]]
  // CHECK: [[IMM1:%[0-9]+]] = clift.imm 1
  // CHECK: [[SUBSCRIPT1:%[0-9]+]] = clift.subscript [[CAST1]], [[IMM1]]
  // CHECK: [[CAST2:%[0-9]+]] = clift.decay [[SUBSCRIPT1]]
  // CHECK: [[IMM2:%[0-9]+]] = clift.imm 2
  // CHECK: [[SUBSCRIPT2:%[0-9]+]] = clift.subscript [[CAST2]], [[IMM2]]
  // CHECK: [[ADDRESSOF2:%[0-9]+]] = clift.addressof [[SUBSCRIPT2]]
  // CHECK: clift.yield [[ADDRESSOF2]] : !clift.ptr<8 to !int32_t>
}
