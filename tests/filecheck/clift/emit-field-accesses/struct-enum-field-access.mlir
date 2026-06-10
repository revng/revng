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
  "/type-definition/1000-CABIFunctionDefinition" as "f" : !void()
>

// Enum type with underlying `int32_t`
!my_enum = !clift.enum<
  "/type-definition/1001-EnumDefinition" as "my_enum" : !int32_t {
    "/enum-entry/1001-EnumDefinition/0" as "A" : 0,
    "/enum-entry/1001-EnumDefinition/1" as "B" : 1
  }
>

// Struct with an `enum` field: the enum should be traversed correctly and
// the underlying type should be reachable through the enum
!s = !clift.struct<
  "/type-definition/1-StructDefinition" : size(8) {
    "/struct-field/1-StructDefinition/0" : offset(0) !int32_t,
    "/struct-field/1-StructDefinition/4" : offset(4) !my_enum
  }
>

module attributes {clift.module} {
  // Access to the enum field at offset 4
  clift.func @test_enum_field<!f>() {
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

  // CHECK-LABEL: clift.func @test_enum_field<!f>
  // CHECK: [[STRUCT:%[0-9]+]] = clift.local : !_type_definition_1_StructDefinition
  // CHECK: [[ADDRESSOF1:%[0-9]+]] = clift.addressof [[STRUCT]] : !clift.ptr<8 to !_type_definition_1_StructDefinition>
  // CHECK: [[ACCESS:%[0-9]+]] = clift.access<indirect 1> [[ADDRESSOF1]]
  // CHECK: [[ADDRESSOF2:%[0-9]+]] = clift.addressof [[ACCESS]]
  // CHECK: [[CAST:%[0-9]+]] = clift.bitcast [[ADDRESSOF2]]
  // CHECK: clift.yield [[CAST]] : !clift.ptr<8 to !int32_t>
}
