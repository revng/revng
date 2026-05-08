#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "limits.h"
#include "stdbool.h"
#include "stddef.h"
#include "stdint.h"

// NOTE: a portable definition of static_assert is provided, but whether the
//       single argument form is valid is implementation-defined. For this
//       reason, the two argument form must always be used where the standard
//       version of the target implementation is not known.

#if __STDC_VERSION__ >= 201112L
#if __STDC_VERSION__ < 202311L
#define static_assert _Static_assert
#endif
#else
#define static_assert(condition, ...) \
  typedef char static_assert_typedef[(condition) ? 1 : 0]
#endif

static_assert(CHAR_MIN == SCHAR_MIN, "CHAR_MIN != SCHAR_MIN");
static_assert(CHAR_MAX == SCHAR_MAX, "CHAR_MAX != SCHAR_MAX");
static_assert(CHAR_MIN == INT8_MIN, "CHAR_MIN != INT8_MIN");
static_assert(CHAR_MAX == INT8_MAX, "CHAR_MAX != INT8_MAX");

//
// Generic
//

typedef uint8_t generic8_t;
typedef uint16_t generic16_t;
typedef uint32_t generic32_t;
typedef uint64_t generic64_t;

#if __SIZEOF_LONG_DOUBLE__ == 10
typedef long double generic80_t;
#else
typedef struct {
  char data[10];
} generic80_t;
#endif

#if __SIZEOF_LONG_DOUBLE__ == 12
typedef long double generic96_t;
#else
typedef struct {
  char data[12];
} generic96_t;
#endif

#ifdef __SIZEOF_INT128__
typedef unsigned __int128 generic128_t;
#endif

static_assert(sizeof(generic8_t) == 1, "");
static_assert(sizeof(generic16_t) == 2, "");
static_assert(sizeof(generic32_t) == 4, "");
static_assert(sizeof(generic64_t) == 8, "");
static_assert(sizeof(generic80_t) == 10, "");
static_assert(sizeof(generic96_t) == 12, "");
#ifdef __SIZEOF_INT128__
static_assert(sizeof(generic128_t) == 16, "");
#endif

//
// PointerOrNumber
//

typedef uint8_t pointer_or_number8_t;
typedef uint16_t pointer_or_number16_t;
typedef uint32_t pointer_or_number32_t;
typedef uint64_t pointer_or_number64_t;
#ifdef __SIZEOF_INT128__
typedef unsigned __int128 pointer_or_number128_t;
#endif

static_assert(sizeof(pointer_or_number8_t) == 1, "");
static_assert(sizeof(pointer_or_number16_t) == 2, "");
static_assert(sizeof(pointer_or_number32_t) == 4, "");
static_assert(sizeof(pointer_or_number64_t) == 8, "");
#ifdef __SIZEOF_INT128__
static_assert(sizeof(pointer_or_number128_t) == 16, "");
#endif

//
// Number
//

typedef uint8_t number8_t;
typedef uint16_t number16_t;
typedef uint32_t number32_t;
typedef uint64_t number64_t;
#ifdef __SIZEOF_INT128__
typedef unsigned __int128 number128_t;
#endif

static_assert(sizeof(number8_t) == 1, "");
static_assert(sizeof(number16_t) == 2, "");
static_assert(sizeof(number32_t) == 4, "");
static_assert(sizeof(number64_t) == 8, "");
#ifdef __SIZEOF_INT128__
static_assert(sizeof(number128_t) == 16, "");
#endif

//
// Signed and Unsigned
//

// Smaller sizes are already present in stdint.h
#ifdef __SIZEOF_INT128__
typedef __int128 int128_t;
typedef unsigned __int128 uint128_t;
#endif

static_assert(sizeof(int8_t) == 1, "");
static_assert(sizeof(int16_t) == 2, "");
static_assert(sizeof(int32_t) == 4, "");
static_assert(sizeof(int64_t) == 8, "");
#ifdef __SIZEOF_INT128__
static_assert(sizeof(int128_t) == 16, "");
#endif

static_assert(sizeof(uint8_t) == 1, "");
static_assert(sizeof(uint16_t) == 2, "");
static_assert(sizeof(uint32_t) == 4, "");
static_assert(sizeof(uint64_t) == 8, "");
#ifdef __SIZEOF_INT128__
static_assert(sizeof(uint128_t) == 16, "");
#endif

//
// Float
//

#if (__ARM_FP16_ARGS == 1 || defined(__FLT16_MIN__)) \
  && !defined(DISABLE_FLOAT16)
typedef _Float16 float16_t;
#else
typedef struct {
  char data[2];
} float16_t;
#endif

#if __SIZEOF_FLOAT__ == 4
typedef float float32_t;
#else
typedef struct {
  char data[4];
} float32_t;
#endif

#if __SIZEOF_DOUBLE__ == 8
typedef double float64_t;
#else
typedef struct {
  char data[8];
} float64_t;
#endif

#if __SIZEOF_LONG_DOUBLE__ == 10
typedef long double float80_t;
#else
typedef struct {
  char data[10];
} float80_t;
#endif

#if __SIZEOF_LONG_DOUBLE__ == 12
typedef long double float96_t;
#else
typedef struct {
  char data[12];
} float96_t;
#endif

#if __SIZEOF_LONG_DOUBLE__ == 16
typedef long double float128_t;
#else
#if defined(__FLT128_MIN__)
typedef _Float128 float128_t;
#else

typedef struct {
  char data[16];
} float128_t;
#endif
#endif

static_assert(sizeof(float16_t) == 2, "");
static_assert(sizeof(float32_t) == 4, "");
static_assert(sizeof(float64_t) == 8, "");
static_assert(sizeof(float80_t) == 10, "");
static_assert(sizeof(float96_t) == 12, "");
static_assert(sizeof(float128_t) == 16, "");

//
// Pointers
//

#ifdef LEGACY_BACKEND
#define pointer16_t(T) __typeof__((T *){ 0 })
#define pointer32_t(T) __typeof__((T *){ 0 })
#define pointer64_t(T) __typeof__((T *){ 0 })
#else
#define pointer16_t(T) uint16_t
#define pointer32_t(T) uint32_t
#define pointer64_t(T) uint64_t
#endif

//
// Undefined values
//

extern uintmax_t undef_value(void);

#define undef(T) ((T) undef_value())

//
// Break and continue
//

#define break_to goto
#define continue_to goto

//
// __typeof__
//

// __typeof__ is required for properly implementing a generic bitcast primitive
// and a generic array_assignment primitive that only need the destination type.
#if defined(__GNUC__) // For Clang and GCC, they both have __typeof__
// Don't do anything, __typeof__ is already available
#elif __STDC_VERSION__ >= 202311L // C23 has typeof
// The macro is variadic so we can pass in expressions that contain commas.
#define __typeof__(...) typeof((__VA_ARGS__))
#else
#error "Neither (C23) typeof nor the __typeof__ compiler extension is available"
#endif

//
// bit_cast
// The macro is variadic so we can pass in expressions that contain commas.
//
#if defined(__GNUC__)

#if defined(__clang__)

#define bit_cast(T, ...) __builtin_bit_cast(T, ((__VA_ARGS__)))

#else // !defined(__clang__), hence, it's GCC

#define bit_cast(T, ...)                                   \
  (__extension__({                                         \
    T bit_cast_r;                                          \
    __typeof__((__VA_ARGS__)) bit_cast_v = (__VA_ARGS__);  \
    __builtin_memcpy(&bit_cast_r, &bit_cast_v, sizeof(T)); \
    bit_cast_r;                                            \
  }))

#endif // defined(__clang__)

#else // !defined(__GNU_C__)

#define bit_cast(T, ...)            \
  (((union {                        \
     __typeof__((__VA_ARGS__)) src; \
     T dst;                         \
   }){ .src = ((__VA_ARGS__)) })    \
     .dst)

#endif // defined(__GNU_C_)

//
// array_assign and array_bit_cast
//

#define assign_array(x, ...)                             \
  (*(__typeof__(x) *) __builtin_memcpy(&(x),             \
                                       &((__VA_ARGS__)), \
                                       sizeof(__typeof__(x))))

// The macro is variadic so we can pass in expressions that contain commas.
#define bit_cast_to_array(element_type, num_elements, ...) \
  (((union {                                               \
     __typeof__((__VA_ARGS__)) src;                        \
     element_type dst[num_elements];                       \
   }){ .src = ((__VA_ARGS__)) })                           \
     .dst)
