#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "primitive-types.h"

//
// Runtime library ABI
//

void imm_impl(void *out, size_t size, char const *literal);

void zext_impl(const void *in, size_t in_size, void *out, size_t out_size);
void sext_impl(const void *in, size_t in_size, void *out, size_t out_size);
void truncate_impl(const void *in, size_t in_size, void *out, size_t out_size);

void neg_impl(void *in_out, size_t size);
void add_impl(void *in_out, const void *in, size_t size);
void sub_impl(void *in_out, const void *in, size_t size);
void mul_impl(void *in_out, const void *in, size_t size);
void sdiv_impl(void *in_out, const void *in, size_t size);
void udiv_impl(void *in_out, const void *in, size_t size);
void srem_impl(void *in_out, const void *in, size_t size);
void urem_impl(void *in_out, const void *in, size_t size);

void shl_impl(void *in_out, const void *in, size_t size);
void shr_impl(void *in_out, const void *in, size_t size);
void sar_impl(void *in_out, const void *in, size_t size);

void bitnot_impl(void *in_out, size_t size);
void bitand_impl(void *in_out, const void *in, size_t size);
void bitor_impl(void *in_out, const void *in, size_t size);
void bitxor_impl(void *in_out, const void *in, size_t size);

void inc_impl(void *in_out, size_t size);
void dec_impl(void *in_out, size_t size);

int test_impl(const void *in, size_t size);
int ecmp_impl(const void *lhs, const void *rhs, size_t size);
int scmp_impl(const void *lhs, const void *rhs, size_t size);
int ucmp_impl(const void *lhs, const void *rhs, size_t size);

//
// Operator implementation macros
//

#define __operator_imm(type, constant)    \
  (__extension__({                        \
    type __r;                             \
    imm_impl(__r, sizeof(__r), constant); \
    __r;                                  \
  }))

#define __operator_cast(operator, type, x)                 \
  (__extension__({                                         \
    __typeof__(x) __x = (x);                               \
    __typeof__((type){ 0 }) __r;                           \
    operator##_impl(&__x, sizeof(__x), &__r, sizeof(__r)); \
    __r;                                                   \
  }))

#define __operator_1(operator, x)       \
  (__extension__({                      \
    __typeof__(x) __x = (x);            \
    operator##_impl(&__x, sizeof(__x)); \
    __x;                                \
  }))

#define __operator_2(operator, x, y)          \
  (__extension__({                            \
    __typeof__(x) __x = (x);                  \
    __typeof__(y) __y = (y);                  \
    operator##_impl(&__x, &__y, sizeof(__x)); \
    __x;                                      \
  }))

#define __operator_inc(operator, x)      \
  (__extension__({                       \
    __typeof__(x) *__x = &(x);           \
    operator##_impl(*__x, sizeof(*__x)); \
    *__x;                                \
  }))

#define __operator_post_inc(operator, x) \
  (__extension__({                       \
    __typeof__(x) *__x = &(x);           \
    __typeof__(*__x) __r = *__x;         \
    operator##_impl(*__x, sizeof(*__x)); \
    __r;                                 \
  }))

#define __operator_test(x)        \
  (__extension__({                \
    __typeof__(x) __x = (x);      \
    test_impl(&__x, sizeof(__x)); \
  }))

#define __operator_cmp(operator, x, y)        \
  (__extension__({                            \
    __typeof__(x) __x = (x);                  \
    __typeof__(y) __y = y;                    \
    operator##_impl(&__x, &__y, sizeof(__x)); \
  }))

//
// User-facing operator macros
//

#define imm(type, x) __operator_imm(type, x)

#define zext(type, x) __operator_cast(zext, type, x)
#define sext(type, x) __operator_cast(sext, type, x)
#define truncate(type, x) __operator_cast(truncate, type, x)

#define neg(x) __operator_1(neg, x)
#define add(x, y) __operator_2(add, x, y)
#define sub(x, y) __operator_2(sub, x, y)
#define mul(x, y) __operator_2(mul, x, y)
#define udiv(x, y) __operator_2(udiv, x, y)
#define sdiv(x, y) __operator_2(sdiv, x, y)
#define srem(x, y) __operator_2(srem, x, y)
#define urem(x, y) __operator_2(urem, x, y)

#define shl(x, y) __operator_2(shl, x, y)
#define shr(x, y) __operator_2(shr, x, y)
#define sar(x, y) __operator_2(sar, x, y)

#define bitnot(x) __operator_1(bitnot, x)
#define bitand(x, y) __operator_2(bitand, x, y)
#define bitor(x, y) __operator_2(bitor, x, y)
#define bitxor(x, y) __operator_2(bitxor, x, y)

#define inc(x) __operator_inc(inc, x)
#define dec(x) __operator_inc(dec, x)

#define post_inc(x) __operator_post_inc(inc, x)
#define post_dec(x) __operator_post_inc(dec, x)

#define test(x) __operator_test(x)

#define eq(x, y) (__operator_cmp(ecmp, x, y) == 1)
#define ne(x, y) (__operator_cmp(ecmp, x, y) == 0)
#define slt(x, y) (__operator_cmp(scmp, x, y) < 0)
#define ult(x, y) (__operator_cmp(ucmp, x, y) < 0)
#define sgt(x, y) (__operator_cmp(scmp, x, y) > 0)
#define ugt(x, y) (__operator_cmp(ucmp, x, y) > 0)
#define sle(x, y) (__operator_cmp(scmp, x, y) <= 0)
#define ule(x, y) (__operator_cmp(ucmp, x, y) <= 0)
#define sge(x, y) (__operator_cmp(scmp, x, y) >= 0)
#define uge(x, y) (__operator_cmp(ucmp, x, y) >= 0)
