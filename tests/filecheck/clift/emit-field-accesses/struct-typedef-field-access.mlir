//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s -emit-field-accesses -canonicalize 2>&1 | FileCheck %s

!void = !clift.void
!generic64_t = !clift.int<generic 8>
!int32_t = !clift.int<signed 4>
!int32_t$ptr = !clift.ptr<8 to !int32_t>

// Generic void function prototype with no argument
!f = !clift.func<
  "1000" as "f" : !void()
>

// Typedef wrapping `int32_t`
!my_int = !clift.typedef<"" as "my_int" : !int32_t>

// Struct with a `typedef` field: the traversal should unwrap the `typedef` and
// still match the underlying type correctly
!s = !clift.struct<
  "1" : size(8) {
    "" : offset(0) !my_int,
    "" : offset(4) !int32_t
  }
>

module attributes {clift.module} {
  // Access to the first field (typedef) at offset 0 with `int32_t` pointer type,
  // through the `typedef`.
  clift.func @test_typedef_field<!f>() {
    %0 = clift.local : !s
    clift.expr {
      %1 = clift.addressof %0 : !clift.ptr<8 to !s>
      %2 = clift.bitcast %1 : !clift.ptr<8 to !s> -> !clift.ptr<8 to !int32_t>
      clift.yield %2 : !int32_t$ptr
    }
  }

  // CHECK-LABEL: clift.func @test_typedef_field<!f>
  // CHECK: [[STRUCT:%[0-9]+]] = clift.local : !_1_
  // CHECK: [[ADDRESSOF1:%[0-9]+]] = clift.addressof [[STRUCT]] : !clift.ptr<8 to !_1_>
  // CHECK: [[ACCESS:%[0-9]+]] = clift.ptr_access<0> [[ADDRESSOF1]]
  // CHECK: [[ADDRESSOF2:%[0-9]+]] = clift.addressof [[ACCESS]]
  // CHECK: [[CAST:%[0-9]+]] = clift.bitcast [[ADDRESSOF2]]
  // CHECK: clift.yield [[CAST]] : !clift.ptr<8 to !int32_t>
}
