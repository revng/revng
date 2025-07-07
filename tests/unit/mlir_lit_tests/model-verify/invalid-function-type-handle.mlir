//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngpipe model-verify-clift %S/model.yml %s /dev/null 2>&1 | FileCheck %s

!void = !clift.primitive<void 0>
!t = !clift.func<"/type-definition/1001-StructDefinition" : !void()>

// CHECK: FunctionType with invalid handle: '/type-definition/1001-StructDefinition'
module attributes {clift.module, clift.test = !t} {}
