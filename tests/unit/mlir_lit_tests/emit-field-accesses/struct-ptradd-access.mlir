//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s -emit-field-accesses -canonicalize 2>&1 | FileCheck %s

!void = !clift.void
!generic64_t = !clift.int<generic 8>
!generic64_t$ptr = !clift.ptr<8 to !generic64_t>
!int32_t = !clift.int<signed 4>
!uint32_t = !clift.int<unsigned 4>
!int32_t$ptr = !clift.ptr<8 to !int32_t>

// Generic void function prototype with no argument
!f = !clift.func<
  "1000" as "f" : !void()
>

!s = !clift.struct<
  "1" : size(12) {
    "" : offset(0) !int32_t,
    "" : offset(4) !int32_t,
    "" : offset(8) !int32_t
  }
>
!s$ptr = !clift.ptr<8 to !s>

// Access, with offset computed with a `ptr_add` operation, to the third field
// of the `struct`

module attributes {clift.module} {
  clift.func @f<!f>() {
    %0 = clift.local : !s
    clift.expr {
      %1 = clift.addressof %0 : !clift.ptr<8 to !s>
      %2 = clift.bitcast %1 : !s$ptr -> !int32_t$ptr
      %3 = clift.imm 2 : !generic64_t
      %4 = clift.ptr_add %2, %3 : (!int32_t$ptr, !generic64_t)
      clift.yield %4 : !int32_t$ptr
    }
  }

  // CHECK-LABEL: clift.func @f<!f>
  // CHECK: [[STRUCT:%[0-9]+]] = clift.local : !_1_
  // CHECK: [[ADDRESSOF1:%[0-9]+]] = clift.addressof [[STRUCT]] : !clift.ptr<8 to !_1_>
  // CHECK: [[ACCESS:%[0-9]+]] = clift.access<indirect 2> [[ADDRESSOF1]]
  // CHECK: [[ADDRESSOF2:%[0-9]+]] = clift.addressof [[ACCESS]]
  // CHECK: clift.yield [[ADDRESSOF2]] : !clift.ptr<8 to !int32_t>

  // Access to the second field  of the struct using a `ptr_add` where the
  // `BasePointer` comes from the offset operand (RHS).
  clift.func @g<!f>() {
    %0 = clift.local : !s
    clift.expr {
      %1 = clift.imm 4 : !generic64_t
      %2 = clift.bitcast %1 : !generic64_t -> !int32_t$ptr
      %3 = clift.addressof %0 : !clift.ptr<8 to !s>
      %4 = clift.bitcast %3 : !s$ptr -> !generic64_t
      %5 = clift.ptr_add %2, %4 : (!int32_t$ptr, !generic64_t)
      clift.yield %5 : !int32_t$ptr
    }
  }

  // CHECK-LABEL: clift.func @g<!f>
  // CHECK: [[STRUCT:%[0-9]+]] = clift.local : !_1_
  // CHECK: [[ADDRESSOF1:%[0-9]+]] = clift.addressof [[STRUCT]] : !clift.ptr<8 to !_1_>
  // CHECK: [[ACCESS:%[0-9]+]] = clift.access<indirect 1> [[ADDRESSOF1]]
  // CHECK: [[ADDRESSOF2:%[0-9]+]] = clift.addressof [[ACCESS]]
  // CHECK: clift.yield [[ADDRESSOF2]] : !clift.ptr<8 to !int32_t>
}
