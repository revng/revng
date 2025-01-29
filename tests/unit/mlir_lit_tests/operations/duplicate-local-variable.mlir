//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

!void = !clift.primitive<VoidKind 0>
!int32_t = !clift.primitive<SignedKind 4>

!f = !clift.defined<#clift.function<
  unique_handle = "/model-type/1000",
  name = "f",
  return_type = !void,
  argument_types = []>>

clift.module {
  clift.func @f<!f>() {
    clift.local !int32_t "x"
    // CHECK: conflicts with another local variable
    clift.local !int32_t "x"
  }
}
