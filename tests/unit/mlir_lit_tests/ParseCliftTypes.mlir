//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//
// RUN: diff <(revng clift-opt %s -o -) <(revng clift-opt %s -o - | revng clift-opt -o -)
!const_int32_t = !clift.primitive<is_const = true, SignedKind 4>
!const_uint32_t = !clift.primitive<is_const = true, UnsignedKind 8>
!int32_t = !clift.primitive<SignedKind 4>
#dc2_ = #clift.enum<id = 40, name = "dc2", underlying_type = !const_uint32_t, fields = [<name = "dc3", raw_value = 20>, <name = "dc4", raw_value = 20>]>
#int_type_def = #clift.typedef<id = 40, name = "int_type_def", underlying_type = !const_int32_t>
!dc2_1 = !clift.defined<#dc2_>
!int_type_def1 = !clift.defined<#int_type_def>
#dc = #clift.struct<id = 4, name = "dc", size = 40, fields = [<offset = 10, name = "dc1", type = !clift.primitive<is_const = true, SignedKind 4>>, <offset = 20, name = "dc2", type = !clift.primitive<SignedKind 4>>]>
#dc1 = #clift.union<id = 4, name = "dc", fields = [<offset = 10, name = "dc1", type = !clift.primitive<is_const = true, SignedKind 4>>, <offset = 20, name = "dc2", type = !clift.primitive<SignedKind 4>>]>
!const_dc = !clift.defined<is_const = true, #dc>
!const_dc1 = !clift.defined<is_const = true, #dc1>
#dc2_2 = #clift.function<id = 40, name = "dc2", return_type = !const_dc, argument_types = [#clift.farg<type = !const_int32_t, name = "dc">]>
!dc2_3 = !clift.defined<#dc2_2>
module {
  %0 = clift.undef !const_int32_t
  %1 = clift.undef !const_dc
  %2 = clift.undef !dc2_3
  %3 = clift.undef !dc2_1
  %4 = clift.undef !int_type_def1
  %5 = clift.undef !clift.array<element_type = !const_dc, elements_count = 40>
  %6 = clift.undef !const_dc1
}
