//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s

!void = !clift.primitive<void 0>

!function = !clift.func<
  "/type-definition/1000-CABIFunctionDefinition" as "f" : !void()
>

!function$ptr = !clift.ptr<8 to !function>

%function = clift.undef : !function
clift.cast<decay> %function : !function -> !function$ptr
