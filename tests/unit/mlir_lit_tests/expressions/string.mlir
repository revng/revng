//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s

!char$const = !clift.const<!clift.primitive<number 1>>
clift.str "hello" : !clift.array<6 x !char$const>
