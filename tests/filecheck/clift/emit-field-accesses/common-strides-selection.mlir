//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s -emit-field-accesses -canonicalize 2>&1 | FileCheck %s

// When two candidate traversals reach the same target type at the same offset
// and with the same size, they tie on the `StartDistance`, `SizeRelation` and
// `TypeDistance` criteria, and the selection is decided by `CommonStrides`: the
// candidate whose array structure matches the array structure the access
// actually walks is preferred. `CommonStrides` counts the common prefix between
// the arrays of the `ArrayPath` the access was re-expressed along and the
// arrays of the candidate traversal.
//
// A `union` overlays a genuine 2-D `uint32_t[2][2]` (field 0, outer stride 8,
// inner stride 4, 16 bytes) with a flat `uint32_t[8]` (field 1, stride 4, 32
// bytes). Both fields reach a `uint32_t`, so for a `uint32_t*` access the two
// traversals tie up to `CommonStrides`. The function that walks the buffer with
// a single stride-4 index prefers the flat array; the one that walks it with a
// nested `stride-8`/`stride-4` pair of indices prefers the 2-D array.

!void = !clift.void
!generic64_t = !clift.int<generic 8>
!uint32_t = !clift.int<unsigned 4>
!uint32_t$ptr = !clift.ptr<8 to !uint32_t>

!f = !clift.func<
  "1000" as "f" : !void(!generic64_t)
>
!ff = !clift.func<
  "1001" as "ff" : !void(!generic64_t, !generic64_t)
>

// A genuine 2-D `uint32_t[2][2]` (outer stride 8, inner stride 4).
!deep_inner = !clift.array<2 x !uint32_t>
!deep = !clift.array<2 x !deep_inner>

// A flat `uint32_t[8]` (stride 4).
!flat = !clift.array<8 x !uint32_t>

!u = !clift.union<
  "1" : {
    "" : !deep,
    "" : !flat
  }
>
!u$ptr = !clift.ptr<8 to !u>

module attributes {clift.module} {

  // Single stride-4 index past the 2-D array's bounds: `base + 16 + 4*i`. The
  // constant 16 is two whole `uint32_t[2]` rows, i.e. one past the 2-D array's
  // outer bound. The access walks a single stride-4 array, so its `ArrayPath`
  // is the flat `uint32_t[8]` and `CommonStrides` prefers the flat traversal,
  // yielding the in-bounds `&flat[i + 4]`. The 2-D traversal ties on every
  // earlier criterion but has zero common strides with the access; picking it
  // would emit `&deep[2][i]`, whose outer index 2 is out of bounds.

  clift.func @prefers_flat<!f>(%arg0 : !generic64_t) {
    %0 = clift.local : !u
    clift.expr {
      %1 = clift.imm 4 : !generic64_t
      %2 = clift.mul %1, %arg0 : !generic64_t
      %3 = clift.imm 16 : !generic64_t
      %4 = clift.add %2, %3 : !generic64_t
      %5 = clift.addressof %0 : !u$ptr
      %6 = clift.bitcast %5 : !u$ptr -> !generic64_t
      %7 = clift.add %6, %4 : !generic64_t
      %8 = clift.bitcast %7 : !generic64_t -> !uint32_t$ptr
      clift.yield %8 : !uint32_t$ptr
    }
  }

  // CHECK-LABEL: clift.func @prefers_flat<!f>
  // CHECK-SAME: ([[ARG0:%[a-z0-9]+]]: !generic64_t)
  // CHECK: clift.access<indirect 1> {{.*}} -> !clift.array<8 x !uint32_t>
  // CHECK: [[DECAY:%[0-9]+]] = clift.decay
  // CHECK: [[FOUR:%[0-9]+]] = clift.imm 4
  // CHECK: [[IDX:%[0-9]+]] = clift.add [[FOUR]], [[ARG0]]
  // CHECK: clift.subscript [[DECAY]], [[IDX]]
  // CHECK-NOT: clift.subscript
  // CHECK-NOT: clift.access<indirect 0>

  // Nested stride-8 / stride-4 indices: `base + 8*i + 4*j`. Now the access
  // walks two arrays whose strides are exactly the 2-D array's outer and inner
  // strides, so its `ArrayPath` is the 2-D `uint32_t[2][2]` and `CommonStrides`
  // prefers the 2-D traversal: `&deep[i][j]`, both coefficients 1. The flat
  // array could also represent the access (as `&flat[2*i + j]`) and ties on
  // every earlier criterion, but shares no array prefix with it.

  clift.func @prefers_nested<!ff>(%i : !generic64_t, %j : !generic64_t) {
    %0 = clift.local : !u
    clift.expr {
      %1 = clift.imm 8 : !generic64_t
      %2 = clift.mul %1, %i : !generic64_t
      %3 = clift.imm 4 : !generic64_t
      %4 = clift.mul %3, %j : !generic64_t
      %5 = clift.add %2, %4 : !generic64_t
      %6 = clift.addressof %0 : !u$ptr
      %7 = clift.bitcast %6 : !u$ptr -> !generic64_t
      %8 = clift.add %7, %5 : !generic64_t
      %9 = clift.bitcast %8 : !generic64_t -> !uint32_t$ptr
      clift.yield %9 : !uint32_t$ptr
    }
  }

  // CHECK-LABEL: clift.func @prefers_nested<!ff>
  // CHECK-SAME: ([[I:%[a-z0-9]+]]: !generic64_t, [[J:%[a-z0-9]+]]: !generic64_t)
  // CHECK: clift.access<indirect 0> {{.*}} -> !clift.array<2 x !clift.array<2 x !uint32_t>>
  // CHECK: [[SUB1:%[0-9]+]] = clift.subscript {{%[0-9]+}}, [[I]]
  // CHECK: [[DECAY:%[0-9]+]] = clift.decay [[SUB1]]
  // CHECK: clift.subscript [[DECAY]], [[J]]
  // CHECK-NOT: clift.mul
  // CHECK-NOT: clift.access<indirect 1>
}
