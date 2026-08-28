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

!my_int = !clift.typedef<"" as "my_int" : !int32_t>
!my_int_ptr = !clift.ptr<8 to !my_int>

// Struct with a typedef field at offset 4. The access targets the `typedef`
// itself, exercising a `Traversal` ending on a `TypedefType`.
!s = !clift.struct<
  "1" : size(8) {
    "" : offset(0) !int32_t,
    "" : offset(4) !my_int
  }
>

module attributes {clift.module} {
  // Access to the typedef field itself
  clift.func @test_typedef_target<!f>() {
    %0 = clift.local : !s
    clift.expr {
      %1 = clift.addressof %0 : !clift.ptr<8 to !s>
      %2 = clift.bitcast %1 : !clift.ptr<8 to !s> -> !clift.ptr<8 to !void>
      %3 = clift.bitcast %2 : !clift.ptr<8 to !void> -> !generic64_t
      %4 = clift.imm 4 : !generic64_t
      %5 = clift.add %3, %4 : !generic64_t
      %6 = clift.bitcast %5 : !generic64_t -> !my_int_ptr
      clift.yield %6 : !my_int_ptr
    }
  }

  // CHECK-LABEL: clift.func @test_typedef_target<!f>
  // CHECK: [[STRUCT:%[0-9]+]] = clift.local : !_1_
  // CHECK: [[ADDRESSOF1:%[0-9]+]] = clift.addressof [[STRUCT]] : !clift.ptr<8 to !_1_>
  // CHECK: [[ACCESS:%[0-9]+]] = clift.ptr_access<1> [[ADDRESSOF1]]
  // CHECK: [[ADDRESSOF2:%[0-9]+]] = clift.addressof [[ACCESS]]
  // CHECK: clift.yield [[ADDRESSOF2]] : !clift.ptr<8 to !my_int>
}
