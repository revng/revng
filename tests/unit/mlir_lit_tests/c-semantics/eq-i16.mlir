//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt --verify-c %s 2>&1 | FileCheck %s

!int16_t = !clift.primitive<SignedKind 2>
!int32_t = !clift.primitive<SignedKind 4>

!f = !clift.defined<#clift.function<
  id = 1001,
  name = "",
  return_type = !clift.primitive<VoidKind 0>,
  argument_types = []>>

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
