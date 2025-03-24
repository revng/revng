//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s

!char$const = !clift.primitive<const number 1>
clift.str "hello" : !clift.array<element_type = !char$const, elements_count = 6>
