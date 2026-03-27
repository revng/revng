#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#ifdef DISABLE_ATTRIBUTES
#define __attribute__(argument)
#endif

// Helper macros
#define _CUSTOM_ATTRIBUTE(value) __attribute__((annotate(#value)))
#define _CUSTOM_ANNOTATION(key, value) \
  __attribute__((annotate(#key ":" #value)))

// Custom attributes
#define _STACK _CUSTOM_ATTRIBUTE(stack)
#define _CAN_CONTAIN_CODE _CUSTOM_ATTRIBUTE(can_contain_code)

// Custom attributes with an argument
#define _REG(x) _CUSTOM_ANNOTATION(reg, x)
#define _ABI(x) _CUSTOM_ANNOTATION(abi, x)
#define _STARTS_AT(x) _CUSTOM_ANNOTATION(field_start_offset, x)
#define _SIZE(x) _CUSTOM_ANNOTATION(struct_size, x)
#define _ENUM_UNDERLYING(x) _CUSTOM_ANNOTATION(enum_underlying_type, x)

// Real attribute aliases
#define _PACKED __attribute__((packed))
#define _ALWAYS_INLINE __attribute__((always_inline))
#define _NORETURN __attribute__((noreturn))
