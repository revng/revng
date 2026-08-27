//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s -emit-field-accesses -canonicalize 2>&1 | FileCheck %s

!void = !clift.void
!generic64_t = !clift.int<generic 8>
!int16_t = !clift.int<signed 2>
!int32_t = !clift.int<signed 4>
!uint32_t = !clift.int<unsigned 4>
!int16_t$ptr = !clift.ptr<8 to !int16_t>
!int32_t$ptr = !clift.ptr<8 to !int32_t>

// Generic void function prototype with no argument
!f = !clift.func<
  "1000" as "f" : !void()
>

!nested_u = !clift.union<
  "1" : {
    "" : !int32_t,
    "" : !int16_t
  }
>

!s = !clift.struct<
  "2" : size(8) {
    "" : offset(0) !int32_t,
    "" : offset(4) !nested_u
  }
>

// Access the first field of the union based on the type (size) of the returned
// pointer

module attributes {clift.module} {
  clift.func @f<!f>() {
    %0 = clift.local : !s
    clift.expr {
      %1 = clift.addressof %0 : !clift.ptr<8 to !s>
      %2 = clift.bitcast %1 : !clift.ptr<8 to !s> -> !clift.ptr<8 to !void>
      %3 = clift.bitcast %2 : !clift.ptr<8 to !void> -> !generic64_t
      %4 = clift.imm 4 : !generic64_t
      %5 = clift.add %3, %4 : !generic64_t
      %6 = clift.bitcast %5 : !generic64_t -> !clift.ptr<8 to !int32_t>
      clift.yield %6 : !int32_t$ptr
    }
  }

  // CHECK-LABEL: clift.func @f<!f>
  // CHECK: [[STRUCT:%[0-9]+]] = clift.local : !_2_
  // CHECK: [[ADDRESSOF1:%[0-9]+]] = clift.addressof [[STRUCT]] : !clift.ptr<8 to !_2_>
  // CHECK: [[ACCESS1:%[0-9]+]] = clift.ptr_access<1> [[ADDRESSOF1]]
  // CHECK: [[ACCESS2:%[0-9]+]] = clift.access<0> [[ACCESS1]]
  // CHECK: [[ADDRESSOF2:%[0-9]+]] = clift.addressof [[ACCESS2]]
  // CHECK: clift.yield [[ADDRESSOF2]] : !clift.ptr<8 to !int32_t>

  // Access the second field of the union based on the type (size) of the returned
  // pointer

  clift.func @g<!f>() {
    %0 = clift.local : !s
    clift.expr {
      %1 = clift.addressof %0 : !clift.ptr<8 to !s>
      %2 = clift.bitcast %1 : !clift.ptr<8 to !s> -> !clift.ptr<8 to !void>
      %3 = clift.bitcast %2 : !clift.ptr<8 to !void> -> !generic64_t
      %4 = clift.imm 4 : !generic64_t
      %5 = clift.add %3, %4 : !generic64_t
      %6 = clift.bitcast %5 : !generic64_t -> !clift.ptr<8 to !int16_t>
      clift.yield %6 : !int16_t$ptr
    }
  }

  // CHECK-LABEL: clift.func @g<!f>
  // CHECK: [[STRUCT:%[0-9]+]] = clift.local : !_2_
  // CHECK: [[ADDRESSOF1:%[0-9]+]] = clift.addressof [[STRUCT]] : !clift.ptr<8 to !_2_>
  // CHECK: [[ACCESS1:%[0-9]+]] = clift.ptr_access<1> [[ADDRESSOF1]]
  // CHECK: [[ACCESS2:%[0-9]+]] = clift.access<1> [[ACCESS1]]
  // CHECK: [[ADDRESSOF2:%[0-9]+]] = clift.addressof [[ACCESS2]]
  // CHECK: clift.yield [[ADDRESSOF2]] : !clift.ptr<8 to !int16_t>

  // Access to the second field of the `struct`, plus a `Leftover` remaining part
  // of the access

  clift.func @h<!f>() {
    %0 = clift.local : !s
    clift.expr {
      %1 = clift.addressof %0 : !clift.ptr<8 to !s>
      %2 = clift.bitcast %1 : !clift.ptr<8 to !s> -> !clift.ptr<8 to !void>
      %3 = clift.bitcast %2 : !clift.ptr<8 to !void> -> !generic64_t
      %4 = clift.imm 5 : !generic64_t
      %5 = clift.add %3, %4 : !generic64_t
      %6 = clift.bitcast %5 : !generic64_t -> !clift.ptr<8 to !int16_t>
      clift.yield %6 : !int16_t$ptr
    }
  }

  // CHECK-LABEL: clift.func @h<!f>
  // CHECK: [[STRUCT:%[0-9]+]] = clift.local : !_2_
  // CHECK: [[ADDRESSOF1:%[0-9]+]] = clift.addressof [[STRUCT]] : !clift.ptr<8 to !_2_>
  // CHECK: [[ACCESS:%[0-9]+]] = clift.ptr_access<1> [[ADDRESSOF1]]
  // CHECK: [[ADDRESSOF2:%[0-9]+]] = clift.addressof [[ACCESS]]
  // CHECK: [[CAST1:%[0-9]+]] = clift.bitcast [[ADDRESSOF2]]
  // CHECK: [[IMM:%[0-9]+]] = clift.imm 1
  // CHECK: [[ADD:%[0-9]+]] = clift.add [[CAST1]], [[IMM]]
  // CHECK: [[CAST2:%[0-9]+]] = clift.bitcast [[ADD]]
  // CHECK: [[CAST3:%[0-9]+]] = clift.bitcast [[CAST2]]
  // CHECK: clift.yield [[CAST3]] : !clift.ptr<8 to !int16_t>
}
