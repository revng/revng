#pragma once

//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
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
typedef __uint128_t generic128_t;
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

extern generic8_t _undef_generic8_t();
extern generic16_t _undef_generic16_t();
extern generic32_t _undef_generic32_t();
extern generic64_t _undef_generic64_t();
extern generic80_t _undef_generic80_t();
extern generic96_t _undef_generic96_t();
#ifdef __SIZEOF_INT128__
extern generic128_t _undef_generic128_t();
#endif

//
// PointerOrNumber
//

typedef uint8_t pointer_or_number8_t;
typedef uint16_t pointer_or_number16_t;
typedef uint32_t pointer_or_number32_t;
typedef uint64_t pointer_or_number64_t;
#ifdef __SIZEOF_INT128__
typedef __uint128_t pointer_or_number128_t;
#endif

static_assert_size(pointer_or_number8_t, 1);
static_assert_size(pointer_or_number16_t, 2);
static_assert_size(pointer_or_number32_t, 4);
static_assert_size(pointer_or_number64_t, 8);
#ifdef __SIZEOF_INT128__
static_assert_size(pointer_or_number128_t, 16);
#endif

extern pointer_or_number8_t undef_pointer_or_number8_t();
extern pointer_or_number16_t undef_pointer_or_number16_t();
extern pointer_or_number32_t undef_pointer_or_number32_t();
extern pointer_or_number64_t undef_pointer_or_number64_t();
#ifdef __SIZEOF_INT128__
extern pointer_or_number128_t undef_pointer_or_number128_t();
#endif

//
// Number
//

typedef uint8_t number8_t;
typedef uint16_t number16_t;
typedef uint32_t number32_t;
typedef uint64_t number64_t;
#ifdef __SIZEOF_INT128__
typedef __uint128_t number128_t;
#endif

static_assert_size(number8_t, 1);
static_assert_size(number16_t, 2);
static_assert_size(number32_t, 4);
static_assert_size(number64_t, 8);
#ifdef __SIZEOF_INT128__
static_assert_size(number128_t, 16);
#endif

extern number8_t undef_number8_t();
extern number16_t undef_number16_t();
extern number32_t undef_number32_t();
extern number64_t undef_number64_t();
#ifdef __SIZEOF_INT128__
extern number128_t undef_number128_t();
#endif

//
// Signed and Unsigned
//

// Smaller sizes are already present in stdint.h
#ifdef __SIZEOF_INT128__
typedef __int128_t int128_t;
typedef __uint128_t uint128_t;
#endif

static_assert_size(int8_t, 1);
static_assert_size(int16_t, 2);
static_assert_size(int32_t, 4);
static_assert_size(int64_t, 8);
#ifdef __SIZEOF_INT128__
static_assert_size(int128_t, 16);
#endif

extern int8_t undef_int8_t();
extern int16_t undef_int16_t();
extern int32_t undef_int32_t();
extern int64_t undef_int64_t();
#ifdef __SIZEOF_INT128__
extern int128_t undef_int128_t();
#endif

static_assert_size(uint8_t, 1);
static_assert_size(uint16_t, 2);
static_assert_size(uint32_t, 4);
static_assert_size(uint64_t, 8);
#ifdef __SIZEOF_INT128__
static_assert_size(uint128_t, 16);
#endif

extern uint8_t undef_uint8_t();
extern uint16_t undef_uint16_t();
extern uint32_t undef_uint32_t();
extern uint64_t undef_uint64_t();
#ifdef __SIZEOF_INT128__
extern uint128_t undef_uint128_t();
#endif

//
// Float
//

#if __ARM_FP16_ARGS == 1 || defined(__FLT16_MIN__)
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

extern float16_t undef_float16_t();
extern float32_t undef_float32_t();
extern float64_t undef_float64_t();
extern float80_t undef_float80_t();
extern float96_t undef_float96_t();
extern float128_t undef_float128_t();

#undef static_assert_size
