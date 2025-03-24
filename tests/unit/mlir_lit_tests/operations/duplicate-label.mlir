//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

!void = !clift.primitive<void 0>

!f = !clift.defined<#clift.func<"/model-type/1000" as "f" : !void()>>

clift.module {
  clift.func @f<!f>() {
    clift.make_label "x"
    // CHECK: conflicts with another label
    clift.make_label "x"
  }
}
