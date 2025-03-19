//
// This file is distributed under the MIT License. See LICENSE.md for details.
//
// RUN: %revngcliftopt %s | %revngcliftopt

!int32_t$const = !clift.primitive<is_const = true, SignedKind 4>
!uint32_t$const = !clift.primitive<is_const = true, UnsignedKind 8>

!my_enum = !clift.defined<#clift.enum<
  unique_handle = "/model-type/1001",
  name = "my_enum",
  underlying_type = !uint32_t$const,
  fields = [
    <
      name = "my_enum_20",
      raw_value = 20
    >,
    <
      name = "my_enum_21",
      raw_value = 21
    >
  ]>>

!my_typedef = !clift.defined<#clift.typedef<
  unique_handle = "/model-type/1002",
  name = "my_typedef",
  underlying_type = !int32_t$const>>

!my_struct$const = !clift.defined<
  is_const = true,
  #clift.struct<
    unique_handle = "/model-type/1003",
    name = "my_struct",
    size = 40,
    fields = [
      <
        offset = 10,
        name = "my_struct_10",
        type = !clift.primitive<
          is_const = true,
          SignedKind 4>
      >,
      <
        offset = 20,
        name = "my_struct_20",
        type = !clift.primitive<SignedKind 4>
      >
    ]>>

!my_union$const = !clift.defined<
  is_const = true,
  #clift.union<
    unique_handle = "/model-type/1004",
    name = "my_union",
    fields = [
      <
        offset = 0,
        name = "my_union_10",
        type = !clift.primitive<
          is_const = true,
          SignedKind 4>
      >,
      <
        offset = 0,
        name = "my_union_20",
        type = !clift.primitive<SignedKind 4>
      >
    ]>>

!my_function = !clift.defined<#clift.function<
  unique_handle = "/model-type/1005",
  name = "my_function",
  return_type = !my_struct$const,
  argument_types = [!int32_t$const]>>

!my_recursive_union = !clift.defined<
  is_const = true,
  #clift.union<
    unique_handle = "/model-type/1006",
    name = "my_recursive_union",
    fields = [
      <
        offset = 0,
        name = "my_recursive_union_10",
        type = !clift.primitive<
          is_const = true,
          SignedKind 4>
      >,
      <
        offset = 0,
        name = "my_recursive_union_20",
        type = !clift.pointer<
          pointee_type = !clift.defined<
            is_const = true,
            #clift.union<unique_handle = "/model-type/1006">>,
          pointer_size = 8>
      >
    ]>>

%0 = clift.undef : !int32_t$const
%1 = clift.undef : !uint32_t$const
%2 = clift.undef : !my_enum
%3 = clift.undef : !my_typedef
%4 = clift.undef : !my_struct$const
%5 = clift.undef : !my_union$const
%6 = clift.undef : !my_function
%7 = clift.undef : !my_recursive_union
