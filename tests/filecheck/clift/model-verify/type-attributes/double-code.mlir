//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not FileCheck < <(%root/bin/revng pipeline run-pipe verify-against-model %S/model.yml <(tar -c --transform 's;.*;/binary;' %s) /dev/null -- --debug-log=model-verify 2>&1)

// CHECK: Duplicate `_CAN_CONTAIN_CODE` attributes found in: '/type-definition/1-StructDefinition'

!s_1 = !clift.struct<"/type-definition/1-StructDefinition" : size(64) {}
[
  #clift.c_attribute<"_CAN_CONTAIN_CODE" : "/macro/_CAN_CONTAIN_CODE">,
  #clift.c_attribute<"_CAN_CONTAIN_CODE" : "/macro/_CAN_CONTAIN_CODE">
]>

module attributes { clift.module, clift.types = [ !s_1 ] } {}
