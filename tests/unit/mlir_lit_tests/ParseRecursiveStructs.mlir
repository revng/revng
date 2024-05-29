//
// This file is distributed under the MIT License. See LICENSE.md for details.
//
// RUN: diff <(%revngcliftopt %s -o -) <(%revngcliftopt %s -o - | %revngcliftopt -o -)
#dc1 = #clift.union<id = 4, name = "dc", fields = [<offset = 10, name = "dc1", type = !clift.primitive<is_const = true, SignedKind 4>>, <offset = 20, name = "dc2", type = !clift.pointer<pointee_type = !clift.defined<is_const = true, #clift.union<id = 4>>, pointer_size = 8>>]>
!const_dc1 = !clift.defined<is_const = true, #dc1>
module {
  %6 = clift.undef !const_dc1
}
