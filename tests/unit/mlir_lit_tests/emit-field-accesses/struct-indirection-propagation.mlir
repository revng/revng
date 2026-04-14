//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s -emit-field-accesses -canonicalize 2>&1 | FileCheck %s

!void = !clift.void
!generic64_t = !clift.int<generic 8>
!int32_t = !clift.int<signed 4>
!int32_t$ptr = !clift.ptr<8 to !int32_t>
!generic64_t$ptr = !clift.ptr<8 to !generic64_t>

// Inner struct: has a field at offset 8 (an int32_t pointer)
!inner = !clift.struct<
  "2" : size(16) {
    "" : offset(0) !generic64_t,
    "" : offset(8) !int32_t$ptr
  }
>
!inner$ptr = !clift.ptr<8 to !inner>

// Outer struct: first field is a pointer to the inner struct
!outer = !clift.struct<
  "1" : size(16) {
    "" : offset(0) !inner$ptr,
    "" : offset(8) !generic64_t
  }
>

!f = !clift.func<"1000" as "f" : !void()>

module attributes {clift.module} {

  // Test the type propagation through an `indirection` barrier
  clift.func @test_indirection_propagation<!f>() {
    %0 = clift.local : !outer
    clift.expr {
      %1 = clift.addressof %0 : !clift.ptr<8 to !outer>
      %2 = clift.bitcast %1 : !clift.ptr<8 to !outer> -> !generic64_t$ptr
      %3 = clift.indirection %2 : !generic64_t$ptr
      %4 = clift.bitcast %3 : !generic64_t -> !generic64_t$ptr
      %5 = clift.imm 1 : !generic64_t
      %6 = clift.ptr_add %4, %5 : (!generic64_t$ptr, !generic64_t)
      clift.yield %6 : !generic64_t$ptr
    }
  }

  // CHECK-LABEL: clift.func @test_indirection_propagation
  // CHECK: [[STRUCT:%[0-9]+]] = clift.local
  // CHECK: [[ADDRESSOF1:%[0-9]+]] = clift.addressof [[STRUCT]] : !clift.ptr<8 to !_1_>
  // CHECK: [[ACCESS1:%[0-9]+]] = clift.access<indirect 0> [[ADDRESSOF1]]
  // CHECK: [[ADDRESSOF2:%[0-9]+]] = clift.addressof [[ACCESS1]]
  // CHECK: [[INDIRECTION:%[0-9]+]] = clift.indirection [[ADDRESSOF2]]
  // CHECK: [[ACCESS2:%[0-9]+]] = clift.access<indirect 1> [[INDIRECTION]]
  // CHECK: [[ADDRESSOF3:%[0-9]+]] = clift.addressof [[ACCESS2]]
  // CHECK: [[CAST:%[0-9]+]] = clift.bitcast [[ADDRESSOF3]]
  // CHECK: clift.yield [[CAST]] : !clift.ptr<8 to !generic64_t>
}
