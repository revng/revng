//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

!void = !clift.primitive<void 0>

// CHECK: parameter type must be an object type
!f = !clift.defined<#clift.func<"/model-type/1000" as "f" : !void(!void)>>
