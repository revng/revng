//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s -emit-field-accesses -canonicalize 2>&1 | FileCheck %s

!void = !clift.primitive<void 0>
!generic64_t = !clift.primitive<generic 8>
!int32_t = !clift.primitive<signed 4>
!int64_t = !clift.primitive<signed 8>
!uint32_t = !clift.primitive<unsigned 4>
!uint64_t = !clift.primitive<unsigned 8>
!int32_t$ptr = !clift.ptr<8 to !int32_t>
!int64_t$ptr = !clift.ptr<8 to !int64_t>

// Generic void function prototype with no argument
!f = !clift.func<
  "1000" as "f" : !void()
>

!s = !clift.struct<
  "1" : size(8) {
    "" : offset(0) !int32_t,
    "" : offset(4) !int32_t
  }
>

!u = !clift.union<
  "2" : {
    "" : !s,
    "" : !int64_t
  }
>

!s2 = !clift.struct<
  "3" : size(12) {
    "" : offset(0) !int32_t,
    "" : offset(4) !u
  }
>

!u2 = !clift.union<
   "4" : {
    "" : !s,
    "" : !int32_t
  }
>

!s3 = !clift.struct<
  "5" : size(12) {
    "" : offset(0) !int32_t,
    "" : offset(4) !u2
  }
>

// Access to the `int64_t` field of the nested union, selected due to the type
// of the access

module attributes {clift.module} {
  clift.func @f<!f>() {
    %0 = clift.local : !s2
    clift.expr {
      %1 = clift.addressof %0 : !clift.ptr<8 to !s2>
      %2 = clift.bitcast %1 : !clift.ptr<8 to !s2> -> !clift.ptr<8 to !void>
      %3 = clift.bitcast %2 : !clift.ptr<8 to !void> -> !generic64_t
      %4 = clift.imm 4 : !generic64_t
      %5 = clift.add %3, %4 : !generic64_t
      %6 = clift.bitcast %5 : !generic64_t -> !clift.ptr<8 to !int64_t>
      clift.yield %6 : !int64_t$ptr
    }
  }

  // CHECK-LABEL: clift.func @f<!f>
  // CHECK: [[STRUCT:%[0-9]+]] = clift.local : !_3_
  // CHECK: [[ADDRESSOF1:%[0-9]+]] = clift.addressof [[STRUCT]] : !clift.ptr<8 to !_3_>
  // CHECK: [[ACCESS1:%[0-9]+]] = clift.access<indirect 1> [[ADDRESSOF1]]
  // CHECK: [[ACCESS2:%[0-9]+]] = clift.access< 1> [[ACCESS1]]
  // CHECK: [[ADDRESSOF2:%[0-9]+]] = clift.addressof [[ACCESS2]]
  // CHECK: clift.yield [[ADDRESSOF2]] : !clift.ptr<8 to !int64_t>

  // Access to the `struct` field of the nested union, selected due to the type of
  // the access towards the nested struct field

  clift.func @g<!f>() {
    %0 = clift.local : !s2
      clift.expr {
      %1 = clift.addressof %0 : !clift.ptr<8 to !s2>
      %2 = clift.bitcast %1 : !clift.ptr<8 to !s2> -> !clift.ptr<8 to !void>
      %3 = clift.bitcast %2 : !clift.ptr<8 to !void> -> !generic64_t
      %4 = clift.imm 4 : !generic64_t
      %5 = clift.add %3, %4 : !generic64_t
      %6 = clift.bitcast %5 : !generic64_t -> !clift.ptr<8 to !int32_t>
      clift.yield %6 : !int32_t$ptr
    }
  }

  // CHECK-LABEL: clift.func @g<!f>
  // CHECK: [[STRUCT:%[0-9]+]] = clift.local : !_3_
  // CHECK: [[ADDRESSOF1:%[0-9]+]] = clift.addressof [[STRUCT]] : !clift.ptr<8 to !_3_>
  // CHECK: [[ACCESS1:%[0-9]+]] = clift.access<indirect 1> [[ADDRESSOF1]]
  // CHECK: [[ACCESS2:%[0-9]+]] = clift.access< 0> [[ACCESS1]]
  // CHECK: [[ACCESS3:%[0-9]+]] = clift.access< 0> [[ACCESS2]]
  // CHECK: [[ADDRESSOF2:%[0-9]+]] = clift.addressof [[ACCESS3]]
  // CHECK: clift.yield [[ADDRESSOF2]] : !clift.ptr<8 to !int32_t>

  // Access to the `struct` field of the nested union, second field, selected due
  // to the offset into the nested `struct`

  clift.func @h<!f>() {
    %0 = clift.local : !s2
      clift.expr {
      %1 = clift.addressof %0 : !clift.ptr<8 to !s2>
      %2 = clift.bitcast %1 : !clift.ptr<8 to !s2> -> !clift.ptr<8 to !void>
      %3 = clift.bitcast %2 : !clift.ptr<8 to !void> -> !generic64_t
      %4 = clift.imm 8 : !generic64_t
      %5 = clift.add %3, %4 : !generic64_t
      %6 = clift.bitcast %5 : !generic64_t -> !clift.ptr<8 to !int32_t>
      clift.yield %6 : !int32_t$ptr
    }
  }

  // CHECK-LABEL: clift.func @h<!f>
  // CHECK: [[STRUCT:%[0-9]+]] = clift.local : !_3_
  // CHECK: [[ADDRESSOF1:%[0-9]+]] = clift.addressof [[STRUCT]] : !clift.ptr<8 to !_3_>
  // CHECK: [[ACCESS1:%[0-9]+]] = clift.access<indirect 1> [[ADDRESSOF1]]
  // CHECK: [[ACCESS2:%[0-9]+]] = clift.access< 0> [[ACCESS1]]
  // CHECK: [[ACCESS3:%[0-9]+]] = clift.access< 1> [[ACCESS2]]
  // CHECK: [[ADDRESSOF2:%[0-9]+]] = clift.addressof [[ACCESS3]]
  // CHECK: clift.yield [[ADDRESSOF2]] : !clift.ptr<8 to !int32_t>


  // Access to the `int32_t` field of the nested union, selected due to the type
  // of the access and the depth of the access

  clift.func @i<!f>() {
    %0 = clift.local : !s3
      clift.expr {
      %1 = clift.addressof %0 : !clift.ptr<8 to !s3>
      %2 = clift.bitcast %1 : !clift.ptr<8 to !s3> -> !clift.ptr<8 to !void>
      %3 = clift.bitcast %2 : !clift.ptr<8 to !void> -> !generic64_t
      %4 = clift.imm 4 : !generic64_t
      %5 = clift.add %3, %4 : !generic64_t
      %6 = clift.bitcast %5 : !generic64_t -> !clift.ptr<8 to !int32_t>
      clift.yield %6 : !int32_t$ptr
    }
  }

  // CHECK-LABEL: clift.func @i<!f>
  // CHECK: [[STRUCT:%[0-9]+]] = clift.local : !_5_
  // CHECK: [[ADDRESSOF1:%[0-9]+]] = clift.addressof [[STRUCT]] : !clift.ptr<8 to !_5_>
  // CHECK: [[ACCESS1:%[0-9]+]] = clift.access<indirect 1> [[ADDRESSOF1]]
  // CHECK: [[ACCESS2:%[0-9]+]] = clift.access< 1> [[ACCESS1]]
  // CHECK: [[ADDRESSOF2:%[0-9]+]] = clift.addressof [[ACCESS2]]
  // CHECK: clift.yield [[ADDRESSOF2]] : !clift.ptr<8 to !int32_t>
}
