//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngpipe verify-against-model %S/model.yml %s /dev/null 2>&1 | FileCheck %s

!void = !clift.void
!t = !clift.func<"/type-definition/1001-StructDefinition" : !void()>

// CHECK: a FunctionType with an invalid handle: '/type-definition/1001-StructDefinition'
module attributes {clift.module, clift.test = !t} {}
