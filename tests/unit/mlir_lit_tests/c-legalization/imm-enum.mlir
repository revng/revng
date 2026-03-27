//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s --c-legalization | FileCheck %s

!void = !clift.void

!int32_t = !clift.int<signed 4>
!uint32_t = !clift.int<unsigned 4>

!int64_t = !clift.int<signed 8>
!uint64_t = !clift.int<unsigned 8>

!f = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" : !void()
>

!my_enum = !clift.enum<
  "/type-definition/2001-EnumDefinition" as "my_enum" : !int32_t {
    "/enum-entry/2001-EnumDefinition/0" as "my_enum_0" : 0
  }
>

module attributes {clift.module} {
  clift.func @f<!f>() attributes {
    handle = "/function/0x40001001:Code_x86_64"
  } {
    clift.expr {
      // CHECK: %0 = clift.imm 0 : !my_enum
      %0 = clift.imm 0 : !my_enum
      // CHECK: clift.yield %0 : !my_enum
      clift.yield %0 : !my_enum
    }

    clift.expr {
      // CHECK: %0 = clift.imm 1 : !int32_t
      %0 = clift.imm 1 : !my_enum
      // CHECK: %1 = clift.bitcast %0 : !int32_t -> !my_enum
      // CHECK: clift.yield %1 : !my_enum
      clift.yield %0 : !my_enum
    }
  }
}
