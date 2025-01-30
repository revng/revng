//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt --verify-c %s 2>&1 | FileCheck %s

!int16_t = !clift.primitive<SignedKind 2>

!f = !clift.defined<#clift.function<
  id = 1001,
  name = "",
  return_type = !clift.primitive<VoidKind 0>,
  argument_types = []>>

clift.module {
  clift.func @f<!f>() {
    // CHECK: is not representable in the target implementation
    clift.expr {
      %0 = clift.imm 0 : !int16_t
      clift.yield %0 : !int16_t
    }
  }
}
