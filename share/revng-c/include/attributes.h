#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#define STR(x) #x
// NOLINTNEXTLINE
// clang-format off
#define REG_ATTRIBUTE_STRING(reg_name) STR(reg:reg_name)
#define ABI_ATTRIBUTE_STRING(abi_name) STR(abi:abi_name)
#define ENUM_ATTRIBUTE_STRING(type_name) STR(enum_underlying_type:type_name)
#define FIELD_START_ATTRIBUTE_STRING(offset) STR(field_start_offset:offset)
// NOLINTNEXTLINE
// clang-format on

#define _REG(x) __attribute__((annotate(REG_ATTRIBUTE_STRING(x))))
#define _ABI(x) __attribute__((annotate(ABI_ATTRIBUTE_STRING(x))))
#define _STACK __attribute__((annotate("stack")))

#define _ENUM_UNDERLYING(x) __attribute__((annotate(ENUM_ATTRIBUTE_STRING(x))))
#define _PACKED __attribute__((packed))
#define _START_AT(x) __attribute__((annotate(FIELD_START_ATTRIBUTE_STRING(x))))
