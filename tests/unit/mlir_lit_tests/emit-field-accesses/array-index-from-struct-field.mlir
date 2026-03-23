//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s -emit-field-accesses -canonicalize 2>&1 | FileCheck %s

!void = !clift.primitive<void 0>
!generic64_t = !clift.primitive<generic 8>
!int32_t = !clift.primitive<signed 4>
!int32_t$ptr = !clift.ptr<8 to !int32_t>

!f = !clift.func<
  "1000" as "f" : !void()
>

!a = !clift.array<10 x !int32_t>

// The following `struct` contains both an `array` and an `index` field. The
// `index` field is used as the dynamic `index` into the `array`.
// The access follows this structure:
// `struct.array[(int64_t)(int32_t) struct.index]`
!s = !clift.struct<
  "1" : size(44) {
    "" : offset(0) !a,
    "" : offset(40) !int32_t
  }
>

module attributes {clift.module} {
  clift.func @test_index_from_field<!f>() {
    %0 = clift.local : !s
    clift.expr {
      // Load the `index` field
      %1 = clift.addressof %0 : !clift.ptr<8 to !s>
      %2 = clift.bitcast %1 : !clift.ptr<8 to !s> -> !clift.ptr<8 to !void>
      %3 = clift.bitcast %2 : !clift.ptr<8 to !void> -> !generic64_t
      %4 = clift.imm 40 : !generic64_t
      %5 = clift.add %3, %4 : !generic64_t
      %6 = clift.bitcast %5 : !generic64_t -> !clift.ptr<8 to !int32_t>
      %7 = clift.indirection %6 : !clift.ptr<8 to !int32_t>
      // Sign-extend to 64 bits
      %8 = clift.extend %7 : !int32_t -> !clift.primitive<signed 8>
      %9 = clift.bitcast %8 : !clift.primitive<signed 8> -> !generic64_t
      // Multiply by element size: `index * 4`
      %10 = clift.imm 4 : !generic64_t
      %11 = clift.mul %9, %10 : !generic64_t
      // Add to array base: `&struct + index * 4`
      %12 = clift.add %3, %11 : !generic64_t
      %13 = clift.bitcast %12 : !generic64_t -> !int32_t$ptr
      clift.yield %13 : !int32_t$ptr
    }
  }

  // The pass resolves both the index load (struct field access) and the array
  // access (struct field + decay + subscript) in a single rewrite:
  // The rewrite emits:
  // 1. `access<indirect 1>` loads the `index` from the `struct` second field
  // 2. `access<indirect 0>` accesses the `array` field in the `struct`
  // 4. `subscript ...` accesses the `array[index]` element
  // CHECK-LABEL: clift.func @test_index_from_field<!f>
  // CHECK: [[LOCAL:%[0-9]+]] = clift.local : !_1_
  // CHECK: [[ADDRESSOF1:%[0-9]+]] = clift.addressof [[LOCAL]] : !clift.ptr<8 to !_1_>
  // CHECK: [[IDX_ACCESS:%[0-9]+]] = clift.access<indirect 1> [[ADDRESSOF1]]
  // CHECK: [[IDX_ADDROF:%[0-9]+]] = clift.addressof [[IDX_ACCESS]]
  // CHECK: [[IDX_LOAD:%[0-9]+]] = clift.indirection [[IDX_ADDROF]]
  // CHECK: [[IDX_EXT:%[0-9]+]] = clift.extend [[IDX_LOAD]]
  // CHECK: [[ARR_ACCESS:%[0-9]+]] = clift.access<indirect 0> [[ADDRESSOF1]]
  // CHECK: [[ARR_DECAY:%[0-9]+]] = clift.decay [[ARR_ACCESS]]
  // CHECK: [[SUBSCRIPT:%[0-9]+]] = clift.subscript [[ARR_DECAY]], [[IDX_EXT]]
  // CHECK: [[ADDRESSOF2:%[0-9]+]] = clift.addressof [[SUBSCRIPT]]
  // CHECK: clift.yield [[ADDRESSOF2]] : !clift.ptr<8 to !int32_t>
}
