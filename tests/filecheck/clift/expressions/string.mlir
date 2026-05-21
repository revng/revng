//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s

!char$const = !clift.const<!clift.int<number 1>>
clift.str "hello" : !clift.array<6 x !char$const>
