//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s

!int32_t = !clift.int<signed 4>

%lvalue = clift.local : !int32_t
clift.inc %lvalue : !int32_t
