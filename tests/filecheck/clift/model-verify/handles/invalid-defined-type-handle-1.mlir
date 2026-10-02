//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not --crash %root/bin/revng pipeline run-pipe verify-against-model %S/model.yml <(%pipe_input %s) /dev/null -- --debug-log=model-verify 2>&1 | FileCheck %s

!s = !clift.struct<"/type-definition/5000-StructDefinition" : size(1) {}>

// CHECK: a DefinedType with an invalid handle: '/type-definition/5000-StructDefinition'
module attributes {clift.module, clift.test = !s} {}
