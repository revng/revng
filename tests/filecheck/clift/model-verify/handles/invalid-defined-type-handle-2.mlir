//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %root/bin/revng pipe verify-against-model %S/model.yml %s /dev/null 2>&1 | FileCheck %s

!s = !clift.struct<"/made-up-kind/something" : size(1) {}>

// CHECK: a DefinedType with an invalid handle: '/made-up-kind/something'
module attributes {clift.module, clift.test = !s} {}
