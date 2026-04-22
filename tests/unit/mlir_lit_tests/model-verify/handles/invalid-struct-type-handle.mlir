//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngpipe verify-against-model %S/model.yml %s /dev/null 2>&1 | FileCheck %s

!t = !clift.struct<"/type-definition/1002-UnionDefinition" : size(1) {}>

// CHECK: a StructType with an invalid handle: '/type-definition/1002-UnionDefinition'
module attributes {clift.module, clift.test = !t} {}
