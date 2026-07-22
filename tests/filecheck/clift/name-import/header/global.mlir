//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// TODO: this test should start *only* from the model, as the clift for it can
//       easily be generated from it.

// RUN: %root/bin/revng pipe import-descriptive-info %S/../model.yml %s /dev/stdout | %root/bin/revng clift-opt | FileCheck %s

!void = !clift.void
!uint8_t = !clift.int<unsigned 1>

!segment = !clift.struct<"/type-definition/2005-StructDefinition" : size(1) {
  "/struct-field/2005-StructDefinition/0" : offset(0) !uint8_t
}>

module attributes { clift.module } {
  // CHECK: clift.global @seg_0x40002001 : !segment attributes {
  // CHECK:   handle = "/segment/0x40002001:Generic64-4"
  // CHECK: }
  clift.global @g : !segment attributes {
    handle = "/segment/0x40002001:Generic64-4"
  }
}
