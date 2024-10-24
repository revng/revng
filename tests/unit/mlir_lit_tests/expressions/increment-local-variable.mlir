//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s

!int32_t = !clift.primitive<SignedKind 4>

%lvalue = clift.local !int32_t "x"
clift.inc %lvalue : !int32_t
