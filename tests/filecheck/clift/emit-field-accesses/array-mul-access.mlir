//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s -emit-field-accesses -canonicalize 2>&1 | FileCheck %s

!void = !clift.void
!generic64_t = !clift.int<generic 8>
!int32_t = !clift.int<signed 4>
!uint32_t = !clift.int<unsigned 4>
!int32_t$ptr = !clift.ptr<8 to !int32_t>

// Generic void function prototype with no argument

!f = !clift.func<
  "1000" as "f" : !void()
>

!a = !clift.array<10 x !int32_t>

!s = !clift.struct<
  "1" : size(8) {
    "" : offset(0) !int32_t,
    "" : offset(4) !int32_t
  }
>

!a2 = !clift.array<2 x !int32_t>

!s2 = !clift.struct<
  "2" : size(12) {
    "" : offset(0) !a2,
    "" : offset(8) !int32_t
  }
>

!a3 = !clift.array<10 x !s>

// `array` access a `mul` operation to compute the offset

module attributes {clift.module} {
  clift.func @f<!f>() {
    %0 = clift.local : !a
    clift.expr {
      %1 = clift.addressof %0 : !clift.ptr<8 to !a>
      %2 = clift.bitcast %1 : !clift.ptr<8 to !a> -> !clift.ptr<8 to !void>
      %3 = clift.bitcast %2 : !clift.ptr<8 to !void> -> !generic64_t
      %4 = clift.imm 4 : !generic64_t
      %5 = clift.imm 2 : !generic64_t
      %6 = clift.mul %4, %5 : !generic64_t
      %7 = clift.add %3, %6 : !generic64_t
      %8 = clift.bitcast %7 : !generic64_t -> !clift.ptr<8 to !int32_t>
      clift.yield %8 : !int32_t$ptr
    }
  }

  // CHECK-LABEL: clift.func @f<!f>
  // CHECK: [[ARRAY:%[0-9]+]] = clift.local : !clift.array<10 x !int32_t>
  // CHECK: [[ADDRESSOF1:%[0-9]+]] = clift.addressof [[ARRAY]] : !clift.ptr<8 to !clift.array<10 x !int32_t>>
  // CHECK: [[INDIRECTION:%[0-9]+]] = clift.indirection [[ADDRESSOF1]]
  // CHECK: [[CAST1:%[0-9]+]] = clift.decay [[INDIRECTION]]
  // CHECK: [[IMM:%[0-9]+]] = clift.imm 2
  // CHECK: [[SUBSCRIPT:%[0-9]+]] = clift.subscript [[CAST1]], [[IMM]]
  // CHECK: [[ADDRESSOF2:%[0-9]+]] = clift.addressof [[SUBSCRIPT]]
  // CHECK: clift.yield [[ADDRESSOF2]] : !clift.ptr<8 to !int32_t>


  // `array` access to the first nested `struct` field

  clift.func @g<!f>() {
    %0 = clift.local : !s2
    clift.expr {
      %1 = clift.addressof %0 : !clift.ptr<8 to !s2>
      %2 = clift.bitcast %1 : !clift.ptr<8 to !s2> -> !clift.ptr<8 to !void>
      %3 = clift.bitcast %2 : !clift.ptr<8 to !void> -> !generic64_t
      %4 = clift.imm 4 : !generic64_t
      %5 = clift.imm 1 : !generic64_t
      %6 = clift.mul %4, %5 : !generic64_t
      %7 = clift.add %3, %6 : !generic64_t
      %8 = clift.bitcast %7 : !generic64_t -> !clift.ptr<8 to !int32_t>
      clift.yield %8 : !int32_t$ptr
    }
  }

  // CHECK-LABEL: clift.func @g<!f>
  // CHECK: [[STRUCT:%[0-9]+]] = clift.local : !_2_
  // CHECK: [[ADDRESSOF1:%[0-9]+]] = clift.addressof [[STRUCT]] : !clift.ptr<8 to !_2_>
  // CHECK: [[ACCESS:%[0-9]+]] = clift.ptr_access<0> [[ADDRESSOF1]]
  // CHECK: [[CAST1:%[0-9]+]] = clift.decay [[ACCESS]]
  // CHECK: [[IMM:%[0-9]+]] = clift.imm 1
  // CHECK: [[SUBSCRIPT:%[0-9]+]] = clift.subscript [[CAST1]], [[IMM]]
  // CHECK: [[ADDRESSOF2:%[0-9]+]] = clift.addressof [[SUBSCRIPT]]
  // CHECK: clift.yield [[ADDRESSOF2]] : !clift.ptr<8 to !int32_t>


  // Access to the second `struct` field nested inside an `array`

  clift.func @h<!f>() {
    %0 = clift.local : !a3
    clift.expr {
      %1 = clift.addressof %0 : !clift.ptr<8 to !a3>
      %2 = clift.bitcast %1 : !clift.ptr<8 to !a3> -> !clift.ptr<8 to !void>
      %3 = clift.bitcast %2 : !clift.ptr<8 to !void> -> !generic64_t
      %4 = clift.imm 8 : !generic64_t
      %5 = clift.imm 2 : !generic64_t
      %6 = clift.mul %4, %5 : !generic64_t
      %7 = clift.add %3, %6 : !generic64_t
      %8 = clift.imm 4 : !generic64_t
      %9 = clift.add %7, %8 : !generic64_t
      %10 = clift.bitcast %9 : !generic64_t -> !clift.ptr<8 to !int32_t>
      clift.yield %10 : !int32_t$ptr
    }
  }

  // CHECK-LABEL: clift.func @h<!f>
  // CHECK: [[ARRAY:%[0-9]+]] = clift.local : !clift.array<10 x !_1_>
  // CHECK: [[ADDRESSOF1:%[0-9]+]] = clift.addressof [[ARRAY]] : !clift.ptr<8 to !clift.array<10 x !_1_>>
  // CHECK: [[INDIRECTION:%[0-9]+]] = clift.indirection [[ADDRESSOF1]]
  // CHECK: [[CAST1:%[0-9]+]] = clift.decay [[INDIRECTION]]
  // CHECK: [[IMM:%[0-9]+]] = clift.imm 2
  // CHECK: [[SUBSCRIPT:%[0-9]+]] = clift.subscript [[CAST1]], [[IMM]]
  // CHECK: [[ACCESS:%[0-9]+]] = clift.access<1> [[SUBSCRIPT]]
  // CHECK: [[ADDRESSOF2:%[0-9]+]] = clift.addressof [[ACCESS]]
  // CHECK: clift.yield [[ADDRESSOF2]] : !clift.ptr<8 to !int32_t>
}
