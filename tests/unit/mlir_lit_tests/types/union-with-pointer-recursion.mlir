//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s | %revngcliftopt

!U1 = !clift.union<"U1" : {
  "x" : !clift.ptr<4 to !clift.union<"U2">>
}>
!U2 = !clift.union<"U2" : {
  "x" : !U1
}>
clift.undef : !U1
