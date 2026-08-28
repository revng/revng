//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s -emit-field-accesses -canonicalize 2>&1 | FileCheck %s

// A pointer walking backward through an object produces a negative stride
// (`idx * -56`) or a negative base offset (`-7`). Neither maps to a field or
// array element access, so `PointerArithmetic::verify()` rejects the result and
// emit-field-accesses leaves the raw pointer arithmetic untouched (no
// `clift.subscript`/`clift.access`/`clift.ptr_access`).

!void = !clift.void
!generic64_t = !clift.int<generic 8>
!uint8_t = !clift.int<unsigned 1>
!uint8_t$ptr = !clift.ptr<8 to !uint8_t>

!f = !clift.func<
  "1000" as "f" : !void()
>

!s = !clift.struct<
  "1" : size(64) {
    "" : offset(0) !generic64_t
  }
>
!s$ptr = !clift.ptr<8 to !s>

module attributes {clift.module} {

  // Negative `Stride`: left as raw pointer arithmetic

  clift.func @f<!f>() {
    %0 = clift.local : !s
    %1 = clift.local : !generic64_t
    clift.expr {
      %2 = clift.imm -56 : !generic64_t
      %3 = clift.mul %1, %2 : !generic64_t
      %4 = clift.addressof %0 : !s$ptr
      %5 = clift.bitcast %4 : !s$ptr -> !uint8_t$ptr
      %6 = clift.ptr_add %5, %3 : (!uint8_t$ptr, !generic64_t)
      clift.yield %6 : !uint8_t$ptr
    }
  }

  // CHECK-LABEL: clift.func @f<!f>
  // CHECK: [[STRUCT:%[0-9]+]] = clift.local : !_1_
  // CHECK: [[INDEX:%[0-9]+]] = clift.local : !generic64_t
  // CHECK: [[STRIDE:%[0-9]+]] = clift.imm -56
  // CHECK: [[OFFSET:%[0-9]+]] = clift.mul [[INDEX]], [[STRIDE]]
  // CHECK: [[ADDRESSOF:%[0-9]+]] = clift.addressof [[STRUCT]] : !clift.ptr<8 to !_1_>
  // CHECK: [[CAST:%[0-9]+]] = clift.bitcast [[ADDRESSOF]] : !clift.ptr<8 to !_1_> -> !clift.ptr<8 to !uint8_t>
  // CHECK: [[PTRADD:%[0-9]+]] = clift.ptr_add [[CAST]], [[OFFSET]]
  // CHECK: clift.yield [[PTRADD]] : !clift.ptr<8 to !uint8_t>
  // CHECK-NOT: clift.subscript
  // CHECK-NOT: clift.access
  // CHECK-NOT: clift.ptr_access

  // Negative `BaseOffset`: left as raw pointer arithmetic

  clift.func @g<!f>() {
    %0 = clift.local : !s
    clift.expr {
      %1 = clift.imm -7 : !generic64_t
      %2 = clift.addressof %0 : !s$ptr
      %3 = clift.bitcast %2 : !s$ptr -> !uint8_t$ptr
      %4 = clift.ptr_add %3, %1 : (!uint8_t$ptr, !generic64_t)
      clift.yield %4 : !uint8_t$ptr
    }
  }

  // CHECK-LABEL: clift.func @g<!f>
  // CHECK: [[STRUCT2:%[0-9]+]] = clift.local : !_1_
  // CHECK: [[BASEOFFSET:%[0-9]+]] = clift.imm -7
  // CHECK: [[ADDRESSOF2:%[0-9]+]] = clift.addressof [[STRUCT2]] : !clift.ptr<8 to !_1_>
  // CHECK: [[CAST2:%[0-9]+]] = clift.bitcast [[ADDRESSOF2]] : !clift.ptr<8 to !_1_> -> !clift.ptr<8 to !uint8_t>
  // CHECK: [[PTRADD2:%[0-9]+]] = clift.ptr_add [[CAST2]], [[BASEOFFSET]]
  // CHECK: clift.yield [[PTRADD2]] : !clift.ptr<8 to !uint8_t>
  // CHECK-NOT: clift.subscript
  // CHECK-NOT: clift.access
  // CHECK-NOT: clift.ptr_access
}
