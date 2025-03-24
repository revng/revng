//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s

!void = !clift.primitive<void 0>
!int32_t = !clift.primitive<signed 4>
!int32_t$const = !clift.primitive<const signed 4>

!f = !clift.defined<#clift.func<
  "/type-definition/1-CABIFunctionDefinition" : !void()
>>

!g = !clift.defined<#clift.func<
  "/type-definition/2-CABIFunctionDefinition" : !void(!int32_t)
>>

!h = !clift.defined<#clift.func<
  "/type-definition/3-CABIFunctionDefinition" : !void(!int32_t$const)
>>

%mi = clift.undef : !int32_t
%ci = clift.undef : !int32_t$const

%g = clift.undef : !g
%h = clift.undef : !h

clift.call %g(%mi) : !g
clift.call %g(%mi : !int32_t) : !g
clift.call %g(%ci : !int32_t$const) : !g

clift.call %h(%mi) : !h
clift.call %h(%mi : !int32_t) : !h
clift.call %h(%ci : !int32_t$const) : !h
