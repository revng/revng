//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngpipe model-verify-clift %S/model.yml %s /dev/null 2>&1 | FileCheck %s

!s = !clift.struct<"/type-definition/5000-StructDefinition" : size(1) {}>

// CHECK: DefinedType with invalid handle: '/type-definition/5000-StructDefinition'
module attributes {clift.module, clift.test = !s} {}
