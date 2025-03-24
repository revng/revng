//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

!void = !clift.primitive<void 0>
!int32_t = !clift.primitive<signed 4>

!f = !clift.defined<#clift.function<
  unique_handle = "/model-type/1000",
  name = "f",
  return_type = !void,
  argument_types = []>>

// CHECK: must be nested within a loop operation
clift.module {
  clift.func @f<!f>() {
    clift.while {
      %0 = clift.undef : !int32_t
      clift.yield %0 : !int32_t
    } {
      clift.switch {
        %0 = clift.undef : !int32_t
        clift.yield %0 : !int32_t
      } default {
        clift.loop_break
      }
    }
  }
}
