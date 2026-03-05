//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s -emit-field-accesses -canonicalize 2>&1 | FileCheck %s

!void = !clift.void
!generic64_t = !clift.int<generic 8>
!int16_t = !clift.int<signed 2>
!int32_t = !clift.int<signed 4>
!int64_t = !clift.int<signed 8>
!uint32_t = !clift.int<unsigned 4>
!int16_t$ptr = !clift.ptr<8 to !int16_t>
!int32_t$ptr = !clift.ptr<8 to !int32_t>

!f = !clift.func<
  "1000" as "f" : !void(!generic64_t)
>

!a = !clift.array<10 x !int32_t>

// Dynamic array access with the index provided as an argument plus a fixed
// offset

module attributes {clift.module} {
  clift.func @f<!f>(%arg0 : !generic64_t) {
    %0 = clift.local : !a

    clift.expr {
      %1 = clift.addressof %0 : !clift.ptr<8 to !a>
      %2 = clift.bitcast %1 : !clift.ptr<8 to !a> -> !clift.ptr<8 to !void>
      %3 = clift.bitcast %2 : !clift.ptr<8 to !void> -> !generic64_t
      %4 = clift.imm 1 : !generic64_t
      %5 = clift.add %4, %arg0 : !generic64_t
      %6 = clift.imm 4 : !generic64_t
      %7 = clift.mul %6, %5 : !generic64_t
      %8 = clift.add %3, %7 : !generic64_t
      %9 = clift.bitcast %8 : !generic64_t -> !clift.ptr<8 to !int32_t>
      clift.yield %9 : !int32_t$ptr
    }
  }

  // TODO: add a unit test going over the bounds of the array with just the constant part of the access index
  // We check that the argument of the function is used as the dynamic index into
  // the array, and a constant argument
  // CHECK: clift.func @f<!f>([[ARG0:%[0-9a-z]*]]: !generic64_t)
  // CHECK: [[ARRAY:%[0-9]+]] = clift.local : !clift.array<10 x !int32_t>
  // CHECK: [[ADDRESSOF1:%[0-9]+]] = clift.addressof [[ARRAY]] : !clift.ptr<8 to !clift.array<10 x !int32_t>>
  // CHECK: [[INDIRECTION:%[0-9]+]] = clift.indirection [[ADDRESSOF1]]
  // CHECK: [[CAST:%[0-9]+]] = clift.decay [[INDIRECTION]]
  // CHECK: [[IMM:%[0-9]+]] = clift.imm 1
  // CHECK: [[ADD:%[0-9]+]] = clift.add [[IMM]], [[ARG0]]
  // CHECK: [[SUBSCRIPT:%[0-9]+]] = clift.subscript [[CAST]], [[ADD]]
  // CHECK: [[ADDRESSOF2:%[0-9]+]] = clift.addressof [[SUBSCRIPT]]
  // CHECK: clift.yield [[ADDRESSOF2]] : !clift.ptr<8 to !int32_t>
}
