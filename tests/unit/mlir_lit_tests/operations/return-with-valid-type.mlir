//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s

!void = !clift.primitive<void 0>
!int32_t = !clift.primitive<signed 4>

!f = !clift.func<
  "/type-definition/1-CABIFunctionDefinition" as "f" : !void()
>

!g = !clift.func<
  "/type-definition/2-CABIFunctionDefinition" : !int32_t(!int32_t)
>

clift.module {
  clift.func @f<!f>() {
    clift.return {}
  }

  clift.func @g<!g>(%arg0 : !int32_t) {
    clift.return {
      clift.yield %arg0 : !int32_t
    }
  }
}
