#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "limits.h"
#include "stdint.h"

#define static_assert_size(TYPE, EXPECTED_SIZE) \
  typedef char static_assertion[sizeof(TYPE) == (EXPECTED_SIZE) ? 1 : -1]

_Static_assert(CHAR_MIN == SCHAR_MIN, "CHAR_MIN != SCHAR_MIN");
_Static_assert(CHAR_MAX == SCHAR_MAX, "CHAR_MAX != SCHAR_MAX");
_Static_assert(CHAR_MIN == INT8_MIN, "CHAR_MIN != INT8_MIN");
_Static_assert(CHAR_MAX == INT8_MAX, "CHAR_MAX != INT8_MAX");

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

static_assert_size(generic8_t, 1);
static_assert_size(generic16_t, 2);
static_assert_size(generic32_t, 4);
static_assert_size(generic64_t, 8);
static_assert_size(generic80_t, 10);
static_assert_size(generic96_t, 12);
#ifdef __SIZEOF_INT128__
static_assert_size(generic128_t, 16);
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

static_assert_size(pointer_or_number8_t, 1);
static_assert_size(pointer_or_number16_t, 2);
static_assert_size(pointer_or_number32_t, 4);
static_assert_size(pointer_or_number64_t, 8);
#ifdef __SIZEOF_INT128__
static_assert_size(pointer_or_number128_t, 16);
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

static_assert_size(number8_t, 1);
static_assert_size(number16_t, 2);
static_assert_size(number32_t, 4);
static_assert_size(number64_t, 8);
#ifdef __SIZEOF_INT128__
static_assert_size(number128_t, 16);
#endif

//
// Signed and Unsigned
//

// Smaller sizes are already present in stdint.h
#ifdef __SIZEOF_INT128__
typedef __int128 int128_t;
typedef unsigned __int128 uint128_t;
#endif

static_assert_size(int8_t, 1);
static_assert_size(int16_t, 2);
static_assert_size(int32_t, 4);
static_assert_size(int64_t, 8);
#ifdef __SIZEOF_INT128__
static_assert_size(int128_t, 16);
#endif

static_assert_size(uint8_t, 1);
static_assert_size(uint16_t, 2);
static_assert_size(uint32_t, 4);
static_assert_size(uint64_t, 8);
#ifdef __SIZEOF_INT128__
static_assert_size(uint128_t, 16);
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

static_assert_size(float16_t, 2);
static_assert_size(float32_t, 4);
static_assert_size(float64_t, 8);
static_assert_size(float80_t, 10);
static_assert_size(float96_t, 12);
static_assert_size(float128_t, 16);

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

#undef static_assert_size

//
// Undefined values
//

extern uintmax_t undef_value(void);

#define undef(T) ((T) undef_value())
