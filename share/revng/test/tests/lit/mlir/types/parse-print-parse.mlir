//
// This file is distributed under the MIT License. See LICENSE.md for details.
//
// RUN: %revngcliftopt %s | %revngcliftopt

!int32_t = !clift.primitive<signed 4>
!int32_t$const = !clift.primitive<const signed 4>

!uint32_t = !clift.primitive<unsigned 8>
!uint32_t$const = !clift.primitive<const unsigned 8>

!my_enum = !clift.defined<#clift.enum<
  "/type-definition/1001-EnumDefinition" as "my_enum" : !uint32_t {
    20 as "my_enum_20",
    21 as "my_enum_21"
  }
>>

!my_typedef = !clift.defined<#clift.typedef<
  "/type-definition/1002-TypedefDefinition" as "my_typedef" : !int32_t$const
>>

!my_struct$const = !clift.defined<const #clift.struct<
  "/type-definition/1003-StructDefinition" as "my_struct" : size(40) {
    offset(10) as "my_struct_10" : !clift.primitive<const signed 4>,
    offset(20) as "my_struct_20" : !clift.primitive<signed 4>
  }
>>

!my_union$const = !clift.defined<const #clift.union<
  "/type-definition/1004-UnionDefinition" as "my_union" : {
    "my_union_10" : !clift.primitive<const signed 4>,
    "my_union_20" : !clift.primitive<signed 4>
  }
>>

!my_function = !clift.defined<#clift.func<
  "/type-definition/1005-CABIFunctionDefinition" as "my_function" : !my_struct$const(!int32_t$const)
>>

!my_recursive_union = !clift.defined<const #clift.union<
  "/type-definition/1006-UnionDefinition" as "my_recursive_union" : {
    "my_recursive_union_10" : !clift.primitive<const signed 4>,
    "my_recursive_union_20" : !clift.ptr<8 to !clift.defined<const #clift.union<"/type-definition/1006-UnionDefinition">>>
  }
>>

%0 = clift.undef : !int32_t$const
%1 = clift.undef : !uint32_t$const
%2 = clift.undef : !my_enum
%3 = clift.undef : !my_typedef
%4 = clift.undef : !my_struct$const
%5 = clift.undef : !my_union$const
%6 = clift.undef : !my_function
%7 = clift.undef : !my_recursive_union
