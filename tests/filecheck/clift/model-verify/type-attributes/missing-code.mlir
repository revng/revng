//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %root/bin/revng pipe verify-against-model %S/model.yml %s /dev/null 2>&1 | FileCheck %s

// CHECK: `_CAN_CONTAIN_CODE` status ('0') does not match the model value ('1') for : '/type-definition/1-StructDefinition'

!s_1 = !clift.struct<"/type-definition/1-StructDefinition" : size(64) {}>

module attributes { clift.module, clift.types = [ !s_1 ] } {}
