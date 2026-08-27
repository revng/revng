//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s -emit-field-accesses -canonicalize 2>&1 | FileCheck %s

// An array index is lowered as a linear combination of the runtime variables:
// each strided term of the offset whose stride is a multiple of the array's
// element stride contributes `(stride / element-stride) * variable` to the
// subscript, and all such terms of one array are summed. A constant offset
// left over after the field/array accesses is emitted as pointer arithmetic; a
// runtime term that no traversed array's stride divides makes the access
// unrepresentable, so it is left as raw pointer arithmetic.

!void = !clift.void
!generic64_t = !clift.int<generic 8>
!uint8_t = !clift.int<unsigned 1>
!uint16_t = !clift.int<unsigned 2>
!uint32_t = !clift.int<unsigned 4>
!uint8_t$ptr = !clift.ptr<8 to !uint8_t>
!uint32_t$ptr = !clift.ptr<8 to !uint32_t>

!f = !clift.func<
  "1000" as "f" : !void(!generic64_t)
>
!ff = !clift.func<
  "1001" as "ff" : !void(!generic64_t, !generic64_t)
>

// A `uint32_t[8]`, element stride 4.
!a8 = !clift.array<8 x !uint32_t>
!a8$ptr = !clift.ptr<8 to !a8>

// A flat `uint32_t[24]`, indexed as a flattened 2-D array.
!a24 = !clift.array<24 x !uint32_t>
!a24$ptr = !clift.ptr<8 to !a24>

// A `uint32_t[4][6]`, a genuine 2-D array (outer stride 24, inner stride 4).
!inner = !clift.array<6 x !uint32_t>
!nested = !clift.array<4 x !inner>
!nested$ptr = !clift.ptr<8 to !nested>

// A `struct` of size 4 with a `uint8_t` at offset 3, as an array element.
!s = !clift.struct<
  "2" : size(4) {
    "" : offset(0) !uint16_t,
    "" : offset(2) !uint8_t,
    "" : offset(3) !uint8_t
  }
>
!sarr = !clift.array<8 x !s>
!sarr$ptr = !clift.ptr<8 to !sarr>
!s24 = !clift.array<24 x !s>
!s24$ptr = !clift.ptr<8 to !s24>

module attributes {clift.module} {

  // ===========================================================================
  // Successful rewrites
  // ===========================================================================

  // Scaled index into an array of structs. An `S[8]` (element stride 4) is
  // walked as `base + 8*i + 3` through a `uint8_t*`. The element stride (4)
  // divides the term stride (8), so the index folds to `[2*i]` and the leftover
  // offset 3 selects the field: `&arr[2*i].field`.

  clift.func @scaled_field<!f>(%arg0 : !generic64_t) {
    %0 = clift.local : !sarr
    clift.expr {
      %1 = clift.imm 8 : !generic64_t
      %2 = clift.mul %1, %arg0 : !generic64_t
      %3 = clift.imm 3 : !generic64_t
      %4 = clift.add %2, %3 : !generic64_t
      %5 = clift.addressof %0 : !sarr$ptr
      %6 = clift.bitcast %5 : !sarr$ptr -> !generic64_t
      %7 = clift.add %6, %4 : !generic64_t
      %8 = clift.bitcast %7 : !generic64_t -> !uint8_t$ptr
      clift.yield %8 : !uint8_t$ptr
    }
  }

  // CHECK-LABEL: clift.func @scaled_field<!f>
  // CHECK-SAME: ([[ARG0:%[a-z0-9]+]]: !generic64_t)
  // CHECK: [[DECAY:%[0-9]+]] = clift.decay
  // CHECK: clift.imm 2
  // CHECK: [[IDX:%[0-9]+]] = clift.mul [[ARG0]], {{%[0-9]+}}
  // CHECK: [[SUB:%[0-9]+]] = clift.subscript [[DECAY]], [[IDX]]
  // CHECK: clift.access<2> [[SUB]]

  // Scaled index plus a constant. A `uint32_t[8]` (element stride 4) walked as
  // `base + 8*i + 12` folds to `&arr[2*i + 3]`.

  clift.func @scaled_plus_constant<!f>(%arg0 : !generic64_t) {
    %0 = clift.local : !a8
    clift.expr {
      %1 = clift.imm 8 : !generic64_t
      %2 = clift.mul %1, %arg0 : !generic64_t
      %3 = clift.imm 12 : !generic64_t
      %4 = clift.add %2, %3 : !generic64_t
      %5 = clift.addressof %0 : !a8$ptr
      %6 = clift.bitcast %5 : !a8$ptr -> !generic64_t
      %7 = clift.add %6, %4 : !generic64_t
      %8 = clift.bitcast %7 : !generic64_t -> !uint32_t$ptr
      clift.yield %8 : !uint32_t$ptr
    }
  }

  // CHECK-LABEL: clift.func @scaled_plus_constant<!f>
  // CHECK-SAME: ([[ARG0:%[a-z0-9]+]]: !generic64_t)
  // CHECK: [[DECAY:%[0-9]+]] = clift.decay
  // CHECK: [[THREE:%[0-9]+]] = clift.imm 3
  // CHECK: [[MUL:%[0-9]+]] = clift.mul [[ARG0]], {{%[0-9]+}}
  // CHECK: [[IDX:%[0-9]+]] = clift.add [[THREE]], [[MUL]]
  // CHECK: clift.subscript [[DECAY]], [[IDX]]

  // Linear combination. A flat `uint32_t[24]` (element stride 4) walked as a
  // flattened 2-D access `base + 24*i + 4*j`: both terms are multiples of 4, so
  // both fold into the same subscript, `&arr[6*i + j]`.

  clift.func @linear_combination<!ff>(%i : !generic64_t, %j : !generic64_t) {
    %0 = clift.local : !a24
    clift.expr {
      %1 = clift.imm 24 : !generic64_t
      %2 = clift.mul %1, %i : !generic64_t
      %3 = clift.imm 4 : !generic64_t
      %4 = clift.mul %3, %j : !generic64_t
      %5 = clift.add %2, %4 : !generic64_t
      %6 = clift.addressof %0 : !a24$ptr
      %7 = clift.bitcast %6 : !a24$ptr -> !generic64_t
      %8 = clift.add %7, %5 : !generic64_t
      %9 = clift.bitcast %8 : !generic64_t -> !uint32_t$ptr
      clift.yield %9 : !uint32_t$ptr
    }
  }

  // CHECK-LABEL: clift.func @linear_combination<!ff>
  // CHECK-SAME: ([[I:%[a-z0-9]+]]: !generic64_t, [[J:%[a-z0-9]+]]: !generic64_t)
  // CHECK: [[DECAY:%[0-9]+]] = clift.decay
  // CHECK: [[MUL:%[0-9]+]] = clift.mul [[I]], {{%[0-9]+}}
  // CHECK: [[IDX:%[0-9]+]] = clift.add [[MUL]], [[J]]
  // CHECK: clift.subscript [[DECAY]], [[IDX]]

  // Nested arrays. The same `base + 24*i + 4*j` over a genuine `uint32_t[4][6]`
  // routes each term to its own dimension (outer stride 24, inner stride 4):
  // `&arr[i][j]`, both coefficients 1.

  clift.func @nested_array<!ff>(%i : !generic64_t, %j : !generic64_t) {
    %0 = clift.local : !nested
    clift.expr {
      %1 = clift.imm 24 : !generic64_t
      %2 = clift.mul %1, %i : !generic64_t
      %3 = clift.imm 4 : !generic64_t
      %4 = clift.mul %3, %j : !generic64_t
      %5 = clift.add %2, %4 : !generic64_t
      %6 = clift.addressof %0 : !nested$ptr
      %7 = clift.bitcast %6 : !nested$ptr -> !generic64_t
      %8 = clift.add %7, %5 : !generic64_t
      %9 = clift.bitcast %8 : !generic64_t -> !uint32_t$ptr
      clift.yield %9 : !uint32_t$ptr
    }
  }

  // CHECK-LABEL: clift.func @nested_array<!ff>
  // CHECK-SAME: ([[I:%[a-z0-9]+]]: !generic64_t, [[J:%[a-z0-9]+]]: !generic64_t)
  // CHECK: [[SUB1:%[0-9]+]] = clift.subscript {{%[0-9]+}}, [[I]]
  // CHECK: [[DECAY:%[0-9]+]] = clift.decay [[SUB1]]
  // CHECK: clift.subscript [[DECAY]], [[J]]
  // CHECK-NOT: clift.mul

  // ===========================================================================
  // A constant offset after the subscript: it selects a field, or is left over
  // ===========================================================================

  // A flat `uint32_t[24]` walked as `base + 24*i + 4*j + 2` through a
  // `uint8_t*`. The linear combination folds to `&arr[6*i + j]`, but the extra
  // two-byte offset lands inside the `uint32_t` element (no sub-field), so it
  // remains as an explicit `+ 2` on the resulting pointer.

  clift.func @constant_leftover<!ff>(%i : !generic64_t, %j : !generic64_t) {
    %0 = clift.local : !a24
    clift.expr {
      %1 = clift.imm 24 : !generic64_t
      %2 = clift.mul %1, %i : !generic64_t
      %3 = clift.imm 4 : !generic64_t
      %4 = clift.mul %3, %j : !generic64_t
      %5 = clift.add %2, %4 : !generic64_t
      %6 = clift.imm 2 : !generic64_t
      %7 = clift.add %5, %6 : !generic64_t
      %8 = clift.addressof %0 : !a24$ptr
      %9 = clift.bitcast %8 : !a24$ptr -> !generic64_t
      %10 = clift.add %9, %7 : !generic64_t
      %11 = clift.bitcast %10 : !generic64_t -> !uint8_t$ptr
      clift.yield %11 : !uint8_t$ptr
    }
  }

  // CHECK-LABEL: clift.func @constant_leftover<!ff>
  // CHECK-SAME: ([[I:%[a-z0-9]+]]: !generic64_t, [[J:%[a-z0-9]+]]: !generic64_t)
  // CHECK: [[MUL:%[0-9]+]] = clift.mul [[I]], {{%[0-9]+}}
  // CHECK: [[IDX:%[0-9]+]] = clift.add [[MUL]], [[J]]
  // CHECK: [[SUB:%[0-9]+]] = clift.subscript {{%[0-9]+}}, [[IDX]]
  // CHECK: [[ADDR:%[0-9]+]] = clift.addressof [[SUB]]
  // CHECK: clift.imm 2
  // CHECK: clift.add
  // CHECK: clift.bitcast {{.*}} -> !clift.ptr<8 to !uint8_t>

  // The same access over an `S[24]` whose element `S` has a `uint8_t` field at
  // offset 2: `base + 24*i + 4*j + 2` folds to `&arr[6*i + j].field` with no
  // leftover, the last access being the field access.

  clift.func @constant_selects_field<!ff>(%i : !generic64_t, %j : !generic64_t) {
    %0 = clift.local : !s24
    clift.expr {
      %1 = clift.imm 24 : !generic64_t
      %2 = clift.mul %1, %i : !generic64_t
      %3 = clift.imm 4 : !generic64_t
      %4 = clift.mul %3, %j : !generic64_t
      %5 = clift.add %2, %4 : !generic64_t
      %6 = clift.imm 2 : !generic64_t
      %7 = clift.add %5, %6 : !generic64_t
      %8 = clift.addressof %0 : !s24$ptr
      %9 = clift.bitcast %8 : !s24$ptr -> !generic64_t
      %10 = clift.add %9, %7 : !generic64_t
      %11 = clift.bitcast %10 : !generic64_t -> !uint8_t$ptr
      clift.yield %11 : !uint8_t$ptr
    }
  }

  // CHECK-LABEL: clift.func @constant_selects_field<!ff>
  // CHECK-SAME: ([[I:%[a-z0-9]+]]: !generic64_t, [[J:%[a-z0-9]+]]: !generic64_t)
  // CHECK: [[MUL:%[0-9]+]] = clift.mul [[I]], {{%[0-9]+}}
  // CHECK: [[IDX:%[0-9]+]] = clift.add [[MUL]], [[J]]
  // CHECK: [[SUB:%[0-9]+]] = clift.subscript {{%[0-9]+}}, [[IDX]]
  // CHECK: [[FIELD:%[0-9]+]] = clift.access<1> [[SUB]]
  // CHECK: clift.addressof [[FIELD]] : !clift.ptr<8 to !uint8_t>
  // CHECK-NOT: clift.imm 2

  // ===========================================================================
  // No rewrite: the access is not representable as an integer index
  // ===========================================================================

  // A `uint32_t[8]` (element stride 4) indexed with a five-byte stride
  // (`base + 5*i`): 4 does not divide 5, no traversed array's stride does, so
  // there is no integer subscript and the raw pointer arithmetic is kept.

  clift.func @non_divisor_stride<!f>(%arg0 : !generic64_t) {
    %0 = clift.local : !a8
    clift.expr {
      %1 = clift.imm 5 : !generic64_t
      %2 = clift.mul %1, %arg0 : !generic64_t
      %3 = clift.addressof %0 : !a8$ptr
      %4 = clift.bitcast %3 : !a8$ptr -> !generic64_t
      %5 = clift.add %4, %2 : !generic64_t
      %6 = clift.bitcast %5 : !generic64_t -> !uint32_t$ptr
      clift.yield %6 : !uint32_t$ptr
    }
  }

  // CHECK-LABEL: clift.func @non_divisor_stride<!f>
  // CHECK: clift.mul
  // CHECK: clift.add
  // CHECK: clift.bitcast {{.*}} -> !clift.ptr<8 to !uint32_t>
  // CHECK-NOT: clift.subscript
  // CHECK-NOT: clift.access
  // CHECK-NOT: clift.ptr_access

  // A `uint32_t[8]` walked as `base + 8*i + 5*j`. `8*i` alone would fold, but
  // `5*j` is a multiple of no array stride, so the whole access is left as raw
  // pointer arithmetic rather than dropping the `j` index.

  clift.func @unrepresentable_term<!ff>(%i : !generic64_t, %j : !generic64_t) {
    %0 = clift.local : !a8
    clift.expr {
      %1 = clift.imm 8 : !generic64_t
      %2 = clift.mul %1, %i : !generic64_t
      %3 = clift.imm 5 : !generic64_t
      %4 = clift.mul %3, %j : !generic64_t
      %5 = clift.add %2, %4 : !generic64_t
      %6 = clift.addressof %0 : !a8$ptr
      %7 = clift.bitcast %6 : !a8$ptr -> !generic64_t
      %8 = clift.add %7, %5 : !generic64_t
      %9 = clift.bitcast %8 : !generic64_t -> !uint32_t$ptr
      clift.yield %9 : !uint32_t$ptr
    }
  }

  // CHECK-LABEL: clift.func @unrepresentable_term<!ff>
  // CHECK: clift.mul
  // CHECK: clift.mul
  // CHECK: clift.add
  // CHECK: clift.bitcast {{.*}} -> !clift.ptr<8 to !uint32_t>
  // CHECK-NOT: clift.subscript
  // CHECK-NOT: clift.access
  // CHECK-NOT: clift.ptr_access
}
