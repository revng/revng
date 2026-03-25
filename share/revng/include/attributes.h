#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#ifdef DISABLE_ATTRIBUTES
#define __attribute__(argument)
#endif

#define _STR(x) #x

// NOLINTNEXTLINE
// clang-format off
// (clang-format is disabled because it breaks formatting around `:`s)

#define _REG(x) __attribute__((annotate(_STR(reg:x))))
#define _ABI(x) __attribute__((annotate(_STR(abi:x))))
#define _STACK __attribute__((annotate(_STR(stack))))

#define _ENUM_UNDERLYING(x) \
  __attribute__((annotate(_STR(enum_underlying_type:x))))
#define _PACKED __attribute__((packed))
#define _CAN_CONTAIN_CODE __attribute__((annotate(_STR(can_contain_code))))
#define _STARTS_AT(x) __attribute__((annotate(_STR(field_start_offset:x))))
#define _SIZE(x) __attribute__((annotate(_STR(struct_size:x))))

// NOLINTNEXTLINE
// clang-format on
