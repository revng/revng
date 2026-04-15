//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngpipe verify-against-model %S/model.yml %s /dev/null

!s_0 = !clift.struct<"/type-definition/0-StructDefinition" : size(64) {}>
!s_1 = !clift.struct<"/type-definition/1-StructDefinition" : size(64) {}
[#clift.c_attribute<"_CAN_CONTAIN_CODE" : "/macro/_CAN_CONTAIN_CODE">]>

module attributes { clift.module, clift.types = [ !s_0, !s_1 ] } {}
