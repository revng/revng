//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s

!int32_t = !clift.primitive<signed 4>
!int64_t = !clift.primitive<signed 8>

!f = !clift.defined<#clift.func<
  "/type-definition/1-CABIFunctionDefinition" : !int32_t()
>>

!g = !clift.defined<#clift.func<
  "/type-definition/2-CABIFunctionDefinition" : !int32_t(!int32_t)
>>

!g$ptr = !clift.ptr<8 to !g>

clift.module {
  clift.func @f<!f>() {
    clift.return {
      %0 = clift.imm 0 : !int64_t
      %g = clift.cast<bitcast> %0 : !int64_t -> !g$ptr
      %i = clift.imm 0 : !int32_t
      %r = clift.call %g(%i) : !g$ptr
      clift.yield %r : !int32_t
    }
  }
}
