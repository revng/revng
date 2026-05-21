//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s -emit-field-accesses -canonicalize 2>&1 | FileCheck %s

!void = !clift.void
!generic64_t = !clift.int<generic 8>
!int32_t = !clift.int<signed 4>

// Generic void function prototype with no argument
!f = !clift.func<
  "1000" as "f" : !void()
>

!int32_ptr = !clift.ptr<8 to !int32_t>
!int32_ptr_ptr = !clift.ptr<8 to !int32_ptr>

// Struct with a pointer field at offset 8. The access targets the pointer
// field itself, exercising a `Traversal` ending on a `PointerType`.
!s = !clift.struct<
  "1" : size(16) {
    "" : offset(0) !int32_t,
    "" : offset(8) !int32_ptr
  }
>

module attributes {clift.module} {
  // Access to the pointer field itself
  clift.func @test_pointer_target<!f>() {
    %0 = clift.local : !s
    clift.expr {
      %1 = clift.addressof %0 : !clift.ptr<8 to !s>
      %2 = clift.bitcast %1 : !clift.ptr<8 to !s> -> !clift.ptr<8 to !void>
      %3 = clift.bitcast %2 : !clift.ptr<8 to !void> -> !generic64_t
      %4 = clift.imm 8 : !generic64_t
      %5 = clift.add %3, %4 : !generic64_t
      %6 = clift.bitcast %5 : !generic64_t -> !int32_ptr_ptr
      clift.yield %6 : !int32_ptr_ptr
    }
  }

  // CHECK-LABEL: clift.func @test_pointer_target<!f>
  // CHECK: [[STRUCT:%[0-9]+]] = clift.local : !_1_
  // CHECK: [[ADDRESSOF1:%[0-9]+]] = clift.addressof [[STRUCT]] : !clift.ptr<8 to !_1_>
  // CHECK: [[ACCESS:%[0-9]+]] = clift.access<indirect 1> [[ADDRESSOF1]]
  // CHECK: [[ADDRESSOF2:%[0-9]+]] = clift.addressof [[ACCESS]]
  // CHECK: clift.yield [[ADDRESSOF2]] : !clift.ptr<8 to !clift.ptr<8 to !int32_t>>
}
