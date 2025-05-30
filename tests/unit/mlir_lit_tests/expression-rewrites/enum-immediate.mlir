//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s --beautify-expressions | FileCheck %s

!void = !clift.primitive<void 0>

!int32_t = !clift.primitive<signed 4>

!my_enum = !clift.enum<"" as "my_enum" : !int32_t { 0 }>
!f = !clift.func<"/model-type/1001" : !void()>

module attributes {clift.module} {
  clift.func @f<!f>() -> !void {
    // CHECK: clift.expr {
    clift.expr {
      // CHECK-NOT: = clift.imm 0 : !int32_t
      %0 = clift.imm 0 : !int32_t
      // CHECK-NOT: clift.cast
      %1 = clift.cast<bitcast> %0 : !int32_t -> !my_enum
      // CHECK: %0 = clift.imm 0 : !my_enum
      // CHECK: clift.yield %0 : !my_enum
      clift.yield %1 : !my_enum
    }
    // CHECK: }
  }
}
