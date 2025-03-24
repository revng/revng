//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s

!void = !clift.primitive<void 0>
!int32_t = !clift.primitive<signed 4>

!f = !clift.defined<#clift.function<
  unique_handle = "/model-type/1",
  name = "f",
  return_type = !void,
  argument_types = []>>

!g = !clift.defined<#clift.function<
  unique_handle = "/model-type/2",
  name = "",
  return_type = !int32_t,
  argument_types = [!int32_t]>>

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
