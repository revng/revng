//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s

!void = !clift.primitive<VoidKind 0>
!int32_t = !clift.primitive<SignedKind 4>

!f = !clift.defined<#clift.function<
  id = 1,
  name = "",
  return_type = !void,
  argument_types = [!int32_t]>>

clift.module {
  clift.func @f<!f>(%arg0 : !int32_t) {
    clift.expr {
      %r = clift.assign %arg0, %arg0 : !int32_t
      clift.yield %r : !int32_t
    }
  }
}
