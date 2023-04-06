#pragma once

//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#define STR(x) #x
// NOLINTNEXTLINE
// clang-format off
#define REG_ATTRIBUTE_STRING(reg_name) STR(reg:reg_name)
#define ABI_ATTRIBUTE_STRING(abi_name) STR(abi:abi_name)
// NOLINTNEXTLINE
// clang-format on
#define REG(x) __attribute__((annotate(REG_ATTRIBUTE_STRING(x))))
#define ABI(x) __attribute__((annotate(ABI_ATTRIBUTE_STRING(x))))
#define STACK __attribute__((annotate("stack")))
