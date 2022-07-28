#pragma once

//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#define static_assert_size(TYPE, EXPECTED_SIZE) \
  typedef char static_assertion[sizeof(TYPE) == (EXPECTED_SIZE) ? 1 : -1]

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
