//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s -emit-field-accesses -canonicalize 2>&1 | FileCheck %s

// A `union` overlays three arrays at the same offset: a `uint32_t[4]` (stride 4,
// field 0), a `uint16_t[8]` (stride 2, field 1) and a `uint8_t[16]` (stride 1,
// field 2). All three functions below walk the buffer with the same two-byte
// byte-stride (`base + 2*i`); the traversal whose target element matches the
// replaced pointer's pointee width scores best (SizeRelation::Same beats Larger
// and Smaller), but only traversals that can represent the two-byte index are
// eligible. A strided term folds into a subscript when the chosen array's
// element stride divides it, scaling the index by the quotient. When the
// best-scoring (size-matching) array cannot represent the index, a lower-scored
// but representable array is chosen instead, with a cast to the requested type.

!void = !clift.void
!generic64_t = !clift.int<generic 8>
!uint8_t = !clift.int<unsigned 1>
!uint16_t = !clift.int<unsigned 2>
!uint32_t = !clift.int<unsigned 4>
!uint8_t$ptr = !clift.ptr<8 to !uint8_t>
!uint16_t$ptr = !clift.ptr<8 to !uint16_t>
!uint32_t$ptr = !clift.ptr<8 to !uint32_t>

!f = !clift.func<
  "1000" as "f" : !void(!generic64_t)
>

!a32 = !clift.array<4 x !uint32_t>
!a16 = !clift.array<8 x !uint16_t>
!a8 = !clift.array<16 x !uint8_t>

!u = !clift.union<
  "1" : {
    "" : !a32,
    "" : !a16,
    "" : !a8
  }
>
!u$ptr = !clift.ptr<8 to !u>

module attributes {clift.module} {

  // Scaled fold. Read through a `uint8_t*`: the best traversal lands on a
  // `uint8_t` element reached through the stride-1 `uint8_t[16]` (field 2). The
  // element stride (1) divides the term stride (2), so the index is scaled by
  // the quotient and folds into `buf[2*i]`.

  clift.func @f<!f>(%arg0 : !generic64_t) {
    %0 = clift.local : !u
    clift.expr {
      %1 = clift.imm 2 : !generic64_t
      %2 = clift.mul %1, %arg0 : !generic64_t
      %3 = clift.addressof %0 : !u$ptr
      %4 = clift.bitcast %3 : !u$ptr -> !uint8_t$ptr
      %5 = clift.ptr_add %4, %2 : (!uint8_t$ptr, !generic64_t)
      clift.yield %5 : !uint8_t$ptr
    }
  }

  // CHECK-LABEL: clift.func @f<!f>
  // CHECK-SAME: ([[ARG0:%[a-z0-9]+]]: !generic64_t)
  // CHECK: clift.access<indirect 2> {{.*}} -> !clift.array<16 x !uint8_t>
  // CHECK: [[DECAY:%[0-9]+]] = clift.decay
  // CHECK: clift.imm 2
  // CHECK: [[IDX:%[0-9]+]] = clift.mul [[ARG0]], {{%[0-9]+}}
  // CHECK: clift.subscript [[DECAY]], [[IDX]]
  // CHECK-NOT: clift.ptr_add

  // Exact fold. Read through a `uint16_t*` (same byte access, `base + 2*i`): the
  // best traversal lands on a `uint16_t` element reached through the stride-2
  // `uint16_t[8]` (field 1). The stride matches the access exactly (quotient 1),
  // so the index folds unscaled into `buf[i]`.

  clift.func @g<!f>(%arg0 : !generic64_t) {
    %0 = clift.local : !u
    clift.expr {
      %1 = clift.addressof %0 : !u$ptr
      %2 = clift.bitcast %1 : !u$ptr -> !uint16_t$ptr
      %3 = clift.ptr_add %2, %arg0 : (!uint16_t$ptr, !generic64_t)
      clift.yield %3 : !uint16_t$ptr
    }
  }

  // CHECK-LABEL: clift.func @g<!f>
  // CHECK-SAME: ([[ARG0:%[a-z0-9]+]]: !generic64_t)
  // CHECK: clift.access<indirect 1> {{.*}} -> !clift.array<8 x !uint16_t>
  // CHECK: [[DECAY:%[0-9]+]] = clift.decay
  // CHECK: clift.subscript [[DECAY]], [[ARG0]]
  // CHECK-NOT: clift.mul
  // CHECK-NOT: clift.ptr_add

  // Fallback selection. Read through a `uint32_t*` (same byte access,
  // `base + 2*i`, built as integer arithmetic so the offset is not rescaled):
  // the size-matching `uint32_t` element is reached through the stride-4
  // `uint32_t[4]` (field 0), but stride 4 does not divide the term stride 2, so
  // that traversal cannot represent the index and is not selected. The next
  // representable traversal wins: the `uint8_t[16]` (field 2) folds the index to
  // `buf[2*i]`, and the result is cast back to `uint32_t*`.

  clift.func @h<!f>(%arg0 : !generic64_t) {
    %0 = clift.local : !u
    clift.expr {
      %1 = clift.imm 2 : !generic64_t
      %2 = clift.mul %1, %arg0 : !generic64_t
      %3 = clift.addressof %0 : !u$ptr
      %4 = clift.bitcast %3 : !u$ptr -> !generic64_t
      %5 = clift.add %4, %2 : !generic64_t
      %6 = clift.bitcast %5 : !generic64_t -> !uint32_t$ptr
      clift.yield %6 : !uint32_t$ptr
    }
  }

  // CHECK-LABEL: clift.func @h<!f>
  // CHECK-SAME: ([[ARG0:%[a-z0-9]+]]: !generic64_t)
  // CHECK-NOT: !clift.array<4 x !uint32_t>
  // CHECK: clift.access<indirect 2> {{.*}} -> !clift.array<16 x !uint8_t>
  // CHECK: [[DECAY:%[0-9]+]] = clift.decay
  // CHECK: [[IDX:%[0-9]+]] = clift.mul [[ARG0]], {{%[0-9]+}}
  // CHECK: clift.subscript [[DECAY]], [[IDX]]
  // CHECK: clift.bitcast {{.*}} -> !clift.ptr<8 to !uint32_t>
}
