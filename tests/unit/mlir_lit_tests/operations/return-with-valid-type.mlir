//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s

!void = !clift.primitive<VoidKind 0>
!int32_t = !clift.primitive<SignedKind 4>

clift.return {
  %0 = clift.undef : !void
  clift.yield %0 : !void
}

// TODO: This is currently invalid. Add support for it.
//  clift.return {
//    %0 = clift.undef : !int32_t
//    clift.yield %0 : !int32_t
//  }
