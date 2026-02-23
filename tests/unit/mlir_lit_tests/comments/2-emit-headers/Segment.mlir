//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: ../%revngpipe emit-type-and-global-header /dev/null %s /dev/null /dev/stdout | ../%revngptml | FileCheck %s

!uint8_t = !clift.primitive<unsigned 1>

// CHECK: /// Unlike functions, segments only support simple comments.
// CHECK: ///
// CHECK: /// To attach extra information to internals, use the corresponding struct
// CHECK: /// type!
// CHECK: uint8_t my_named_segment[64];

module attributes {clift.module} {
  clift.global @my_named_segment : !clift.array<64 x !uint8_t>
  attributes {
    clift.comment = "Unlike functions, segments only support simple comments.\0A\0ATo attach extra information to internals, use the corresponding struct\0Atype!",
    handle = "/segment/0x4:Generic64-64"
  }
}
