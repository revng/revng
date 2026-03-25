#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#ifdef DISABLE_ATTRIBUTES
#define __attribute__(argument)
#endif

#define _CUSTOM_ATTRIBUTE(value) __attribute__((annotate(#value)))
#define _CUSTOM_ANNOTATION(key, value) \
  __attribute__((annotate(#key ":" #value)))

#define _REG(x) _CUSTOM_ANNOTATION(reg, x)
#define _ABI(x) _CUSTOM_ANNOTATION(abi, x)
#define _STACK _CUSTOM_ATTRIBUTE(stack)

#define _ENUM_UNDERLYING(x) _CUSTOM_ANNOTATION(enum_underlying_type, x)
#define _PACKED __attribute__((packed))
#define _CAN_CONTAIN_CODE _CUSTOM_ATTRIBUTE(can_contain_code)
#define _STARTS_AT(x) _CUSTOM_ANNOTATION(field_start_offset, x)
#define _SIZE(x) _CUSTOM_ANNOTATION(struct_size, x)
