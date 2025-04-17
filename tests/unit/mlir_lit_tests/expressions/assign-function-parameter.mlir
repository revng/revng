//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s

!void = !clift.primitive<void 0>
!int32_t = !clift.primitive<signed 4>

!f = !clift.func<
  "/type-definition/1-CABIFunctionDefinition" : !void(!int32_t)
>

module attributes {clift.module} {
  clift.func @f<!f>(%arg0 : !int32_t) {
    clift.expr {
      %r = clift.assign %arg0, %arg0 : !int32_t
      clift.yield %r : !int32_t
    }
  }
}
