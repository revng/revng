//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt --verify-c %s 2>&1 | FileCheck %s

!void = !clift.primitive<void 0>
!int16_t = !clift.primitive<signed 2>
!int32_t = !clift.primitive<signed 4>

!f = !clift.defined<#clift.func<
  "/type-definition/1001-CABIFunctionDefinition" : !void()
>>

clift.module {
  clift.func @f<!f>() {
    // CHECK: not yielding the canonical boolean type
    clift.expr {
      %0 = clift.imm 0 : !int32_t
      %1 = clift.imm 0 : !int32_t
      %2 = clift.eq %0, %1 : !int32_t -> !int16_t
      clift.yield %2 : !int16_t
    }
  }
}
