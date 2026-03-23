//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s -emit-field-accesses -canonicalize 2>&1 | FileCheck %s

!void = !clift.primitive<void 0>
!generic64_t = !clift.primitive<generic 8>
!int32_t = !clift.primitive<signed 4>
!int32_t$ptr = !clift.ptr<8 to !int32_t>

// Generic void function prototype with single argument used for the access
!f = !clift.func<
  "1000" as "f" : !void(!generic64_t)
>

!a = !clift.array<10 x !int32_t>

// Dynamic array access via `ptr_add` with a variable `Offset` operand.
// This is the `composePtrAdd` counterpart of the `composeAdd` test in
// `array-linearcombination-access-argument.mlir`.

module attributes {clift.module} {
  clift.func @test_ptradd_linear_combination<!f>(%arg0 : !generic64_t) {
    %0 = clift.local : !a
    clift.expr {
      %1 = clift.addressof %0 : !clift.ptr<8 to !a>
      %2 = clift.bitcast %1 : !clift.ptr<8 to !a> -> !int32_t$ptr
      %3 = clift.ptr_add %2, %arg0 : (!int32_t$ptr, !generic64_t)
      clift.yield %3 : !int32_t$ptr
    }
  }

  // CHECK-LABEL: clift.func @test_ptradd_linear_combination<!f>
  // CHECK: [[ARRAY:%[0-9]+]] = clift.local : !clift.array<10 x !int32_t>
  // CHECK: [[ADDRESSOF1:%[0-9]+]] = clift.addressof [[ARRAY]] : !clift.ptr<8 to !clift.array<10 x !int32_t>>
  // CHECK: [[INDIRECTION:%[0-9]+]] = clift.indirection [[ADDRESSOF1]]
  // CHECK: [[CAST:%[0-9]+]] = clift.decay [[INDIRECTION]]
  // CHECK: [[SUBSCRIPT:%[0-9]+]] = clift.subscript [[CAST]], [[ARG0:%[a-z0-9]+]]
  // CHECK: [[ADDRESSOF2:%[0-9]+]] = clift.addressof [[SUBSCRIPT]]
  // CHECK: clift.yield [[ADDRESSOF2]] : !clift.ptr<8 to !int32_t>
}
