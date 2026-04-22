//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngpipe verify-against-model %S/model.yml %s /dev/null 2>&1 | FileCheck %s

// CHECK: `_CAN_CONTAIN_CODE` status ('1') does not match the model value ('0') for : '/type-definition/0-StructDefinition'

!s_0 = !clift.struct<"/type-definition/0-StructDefinition" : size(64) {}
[#clift.c_attribute<"_CAN_CONTAIN_CODE" : "/macro/_CAN_CONTAIN_CODE">]>

module attributes { clift.module, clift.types = [ !s_0 ] } {}
