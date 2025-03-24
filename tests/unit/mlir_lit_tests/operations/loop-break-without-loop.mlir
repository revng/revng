//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

!void = !clift.primitive<void 0>

!f = !clift.defined<#clift.func<"/model-type/1000" as "f" : !void()>>

// CHECK: must be nested within a loop operation
clift.module {
  clift.func @f<!f>() {
    clift.loop_break
  }
}
