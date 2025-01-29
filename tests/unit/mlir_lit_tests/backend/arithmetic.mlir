//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s --emit-c="tagless model=%S/model.yml" -o /dev/null | FileCheck %s

!void = !clift.primitive<VoidKind 0>

!int32_t = !clift.primitive<SignedKind 4>

!f = !clift.defined<#clift.function<
  unique_handle = "/model-type/1001",
  name = "",
  return_type = !void,
  argument_types = []>>

clift.module {
  // CHECK: void fun_0x40001001(void) {
  clift.func @f<!f>() attributes {
    unique_handle = "/function/0x40001001:Code_x86_64"
  } {
    // (0 + 1 - 2) * 3 / 4 % 5;
    clift.expr {
      %0 = clift.imm 0 : !int32_t
      %1 = clift.imm 1 : !int32_t
      %2 = clift.imm 2 : !int32_t
      %3 = clift.imm 3 : !int32_t
      %4 = clift.imm 4 : !int32_t
      %5 = clift.imm 5 : !int32_t

      %a = clift.add %0, %1 : !int32_t
      %b = clift.sub %a, %2 : !int32_t
      %c = clift.mul %b, %3 : !int32_t
      %d = clift.div %c, %4 : !int32_t
      %e = clift.rem %d, %5 : !int32_t

      clift.yield %e : !int32_t
    }

    // 0 + (1 - 2) * (3 / (4 % 5)
    clift.expr {
      %0 = clift.imm 0 : !int32_t
      %1 = clift.imm 1 : !int32_t
      %2 = clift.imm 2 : !int32_t
      %3 = clift.imm 3 : !int32_t
      %4 = clift.imm 4 : !int32_t
      %5 = clift.imm 5 : !int32_t

      %b = clift.sub %1, %2 : !int32_t
      %e = clift.rem %4, %5 : !int32_t
      %d = clift.div %3, %e : !int32_t
      %c = clift.mul %b, %d : !int32_t
      %a = clift.add %0, %c : !int32_t

      clift.yield %a : !int32_t
    }

    // 0 % 1 / 2 * 3 + 4 - 5;
    clift.expr {
      %0 = clift.imm 0 : !int32_t
      %1 = clift.imm 1 : !int32_t
      %2 = clift.imm 2 : !int32_t
      %3 = clift.imm 3 : !int32_t
      %4 = clift.imm 4 : !int32_t
      %5 = clift.imm 5 : !int32_t

      %a = clift.rem %0, %1 : !int32_t
      %b = clift.div %a, %2 : !int32_t
      %c = clift.mul %b, %3 : !int32_t
      %d = clift.sub %c, %4 : !int32_t
      %e = clift.add %d, %5 : !int32_t

      clift.yield %e : !int32_t
    }
  }
  // CHECK: }
}
