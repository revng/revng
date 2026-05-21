//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s -emit-field-accesses -canonicalize 2>&1 | FileCheck %s

!void = !clift.void
!generic64_t = !clift.int<generic 8>
!int32_t = !clift.int<signed 4>
!int32_t$ptr = !clift.ptr<8 to !int32_t>

// Generic void function prototype with single argument used as array index
!f = !clift.func<
  "1000" as "f" : !void(!generic64_t)
>

!a = !clift.array<10 x !int32_t>

// Dynamic `array` access with both a variable and a constant index component.
// The access pattern is: base + 4*arg0 + 8, which represents array[arg0 + 2].
// This exercises the `IndexConstantComponent` path in `getExplicitArithmetic`,
// where the `BaseOffse`t (8) is divided by the Stride (4) to produce a constant
// index component (2) alongside the variable component (arg0).

module attributes {clift.module} {
  clift.func @test_index_constant<!f>(%arg0 : !generic64_t) {
    %0 = clift.local : !a
    clift.expr {
      %1 = clift.addressof %0 : !clift.ptr<8 to !a>
      %2 = clift.bitcast %1 : !clift.ptr<8 to !a> -> !clift.ptr<8 to !void>
      %3 = clift.bitcast %2 : !clift.ptr<8 to !void> -> !generic64_t
      %4 = clift.imm 4 : !generic64_t
      %5 = clift.mul %4, %arg0 : !generic64_t
      %6 = clift.imm 8 : !generic64_t
      %7 = clift.add %5, %6 : !generic64_t
      %8 = clift.add %3, %7 : !generic64_t
      %9 = clift.bitcast %8 : !generic64_t -> !int32_t$ptr
      clift.yield %9 : !int32_t$ptr
    }
  }

  // The constant component (2) and variable component (arg0) should both appear
  // in the subscript index as an add expression
  // CHECK: clift.func @test_index_constant<!f>([[ARG0:%[0-9a-z]*]]: !generic64_t)
  // CHECK: [[ARRAY:%[0-9]+]] = clift.local : !clift.array<10 x !int32_t>
  // CHECK: [[ADDRESSOF1:%[0-9]+]] = clift.addressof [[ARRAY]] : !clift.ptr<8 to !clift.array<10 x !int32_t>>
  // CHECK: [[INDIRECTION:%[0-9]+]] = clift.indirection [[ADDRESSOF1]]
  // CHECK: [[CAST:%[0-9]+]] = clift.decay [[INDIRECTION]]
  // CHECK: [[CONST:%[0-9]+]] = clift.imm 2
  // CHECK: [[INDEX:%[0-9]+]] = clift.add [[CONST]], [[ARG0]]
  // CHECK: [[SUBSCRIPT:%[0-9]+]] = clift.subscript [[CAST]], [[INDEX]]
  // CHECK: [[ADDRESSOF2:%[0-9]+]] = clift.addressof [[SUBSCRIPT]]
  // CHECK: clift.yield [[ADDRESSOF2]] : !clift.ptr<8 to !int32_t>
}
