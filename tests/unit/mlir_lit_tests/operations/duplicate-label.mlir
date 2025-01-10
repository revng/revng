//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

!void = !clift.primitive<VoidKind 0>

!f = !clift.defined<#clift.function<
  id = 1000,
  name = "f",
  return_type = !void,
  argument_types = []>>

clift.module {
  clift.func @f<!f>() {
    clift.make_label "x"
    // CHECK: conflicts with another label
    clift.make_label "x"
  }
}
