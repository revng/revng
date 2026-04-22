//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngpipe verify-against-model %S/model.yml %s /dev/null 2>&1 | FileCheck %s

// CHECK: `_CAN_CONTAIN_CODE` attribute must not have any arguments. See '/type-definition/1-StructDefinition'

!s_1 = !clift.struct<"/type-definition/1-StructDefinition" : size(64) {}
[#clift.c_attribute<"_CAN_CONTAIN_CODE" : "/macro/_CAN_CONTAIN_CODE" [42]>]>

module attributes { clift.module, clift.types = [ !s_1 ] } {}
