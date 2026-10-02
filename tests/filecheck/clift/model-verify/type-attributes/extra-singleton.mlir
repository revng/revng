//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not --crash %root/bin/revng pipeline run-pipe verify-against-model %S/model.yml <(%pipe_input %s) /dev/null -- --debug-log=model-verify 2>&1 | FileCheck %s

// CHECK: `_SINGLETON` status ('1') does not match the model value ('0') for : '/type-definition/0-StructDefinition'

!s_0 = !clift.struct<"/type-definition/0-StructDefinition" : size(64) {}
[#clift.c_attribute<"_SINGLETON" : "/macro/_SINGLETON">]>

module attributes { clift.module, clift.types = [ !s_0 ] } {}
