//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s

!void = !clift.primitive<void 0>

!f = !clift.defined<#clift.func<
  "/type-definition/1000-CABIFunctionDefinition" as "f" : !void()
>>
