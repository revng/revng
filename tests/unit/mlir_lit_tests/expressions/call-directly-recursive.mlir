//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s

!int32_t = !clift.primitive<signed 4>

!f = !clift.defined<#clift.func<
  "/type-definition/1-CABIFunctionDefinition" : !int32_t(!int32_t)
>>

clift.module {
  clift.func @f<!f>(%arg : !int32_t) -> !int32_t {
    clift.return {
      %f = clift.use @f : !f
      %r = clift.call %f(%arg) : !f
      clift.yield %r : !int32_t
    }
  }
}
