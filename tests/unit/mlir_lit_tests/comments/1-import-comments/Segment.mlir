//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: ../%revngpipe import-descriptive-info %S/../0-import-types/Segment.yml %s /dev/stdout | ../%revngcliftopt | FileCheck %s

!uint8_t = !clift.int<unsigned 1>

// CHECK: module attributes {clift.module} {
// CHECK:   clift.global @my_named_segment : !clift.array<64 x !uint8_t>
// CHECK:   attributes {
// CHECK:     clift.comment = "Unlike functions, segments only support simple comments.\0A\0ATo attach extra information to internals, use the corresponding struct\0Atype!",
// CHECK:     handle = "/segment/0x4:Generic64-64"
// CHECK:   }
// CHECK: }

module attributes {clift.module} {
  clift.global @"0x4:Generic64-64" : !clift.array<64 x !uint8_t>
  attributes {
    handle = "/segment/0x4:Generic64-64"
  }
}
