//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s

!char = !clift.int<signed 1>

clift.str "hello" : !clift.const<!clift.array<6 x !clift.typedef<"/type-definition/1001-TypedefDefinition" : !char>>>
clift.str "hello" : !clift.array<6 x !clift.typedef<"/type-definition/1002-TypedefDefinition" : !clift.const<!char>>>
clift.str "hello" : !clift.const<!clift.typedef<"/type-definition/1003-TypedefDefinition" : !clift.array<6 x !char>>>
clift.str "hello" : !clift.typedef<"/type-definition/1004-TypedefDefinition" : !clift.const<!clift.array<6 x !char>>>
