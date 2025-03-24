//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

!void = !clift.primitive<void 0>

!f = !clift.defined<#clift.func<"/model-type/1001" as "f" : !void()>>

// CHECK: parameter type must be an object type
!g = !clift.defined<#clift.func<"/model-type/1002" as "g" : !void(!f)>>
