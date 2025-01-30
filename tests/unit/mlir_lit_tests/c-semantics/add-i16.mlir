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
    %1 = clift.local !int16_t "x"

    // CHECK: causes integer promotion in the target implementation
    clift.expr {
      %2 = clift.add %1, %1 : !int16_t
      clift.yield %2 : !int16_t
    }
  }
}
