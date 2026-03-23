//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s -emit-field-accesses -canonicalize 2>&1 | FileCheck %s

!void = !clift.primitive<void 0>
!generic64_t = !clift.primitive<generic 8>
!int32_t = !clift.primitive<signed 4>
!uint32_t = !clift.primitive<unsigned 4>
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

// Access, with offset computed with a `shl` operation, to the third field of
// the `struct`

module attributes {clift.module} {
  clift.func @f<!f>() {
    %0 = clift.local : !s
    clift.expr {
      %1 = clift.addressof %0 : !clift.ptr<8 to !s>
      %2 = clift.bitcast %1 : !clift.ptr<8 to !s> -> !clift.ptr<8 to !void>
      %3 = clift.bitcast %2 : !clift.ptr<8 to !void> -> !generic64_t
      %4 = clift.imm 4 : !generic64_t
      %5 = clift.add %3, %4 : !generic64_t
      %6 = clift.imm 1 : !generic64_t
      %7 = clift.shl %5, %6 : !generic64_t
      %8 = clift.add %3, %7 : !generic64_t
      %9 = clift.bitcast %8 : !generic64_t -> !clift.ptr<8 to !int32_t>
      clift.yield %9 : !int32_t$ptr
    }
  }

  // CHECK-LABEL: clift.func @f<!f>
  // CHECK: [[STRUCT:%[0-9]+]] = clift.local : !_1_
  // CHECK: [[ADDRESSOF1:%[0-9]+]] = clift.addressof [[STRUCT]] : !clift.ptr<8 to !_1_>
  // CHECK: [[ACCESS:%[0-9]+]] = clift.access<indirect 2> [[ADDRESSOF1]]
  // CHECK: [[ADDRESSOF2:%[0-9]+]] = clift.addressof [[ACCESS]]
  // CHECK: clift.yield [[ADDRESSOF2]] : !clift.ptr<8 to !int32_t>
}
