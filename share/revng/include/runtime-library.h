#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "primitive-types.h"

//
// Runtime library ABI
//

void rr_imm_impl(void *out, size_t size, char const *literal);

void rr_zext_impl(const void *in, size_t in_size, void *out, size_t out_size);
void rr_sext_impl(const void *in, size_t in_size, void *out, size_t out_size);
void rr_truncate_impl(const void *in,
                      size_t in_size,
                      void *out,
                      size_t out_size);

void rr_neg_impl(void *in_out, size_t size);
void rr_add_impl(void *in_out, const void *in, size_t size);
void rr_sub_impl(void *in_out, const void *in, size_t size);
void rr_mul_impl(void *in_out, const void *in, size_t size);
void rr_sdiv_impl(void *in_out, const void *in, size_t size);
void rr_udiv_impl(void *in_out, const void *in, size_t size);
void rr_srem_impl(void *in_out, const void *in, size_t size);
void rr_urem_impl(void *in_out, const void *in, size_t size);

void rr_shl_impl(void *in_out, const void *in, size_t size);
void rr_shr_impl(void *in_out, const void *in, size_t size);
void rr_sar_impl(void *in_out, const void *in, size_t size);

void rr_bitnot_impl(void *in_out, size_t size);
void rr_bitand_impl(void *in_out, const void *in, size_t size);
void rr_bitor_impl(void *in_out, const void *in, size_t size);
void rr_bitxor_impl(void *in_out, const void *in, size_t size);

void rr_inc_impl(void *in_out, size_t size);
void rr_dec_impl(void *in_out, size_t size);

int rr_test_impl(const void *in, size_t size);
int rr_ecmp_impl(const void *lhs, const void *rhs, size_t size);
int rr_scmp_impl(const void *lhs, const void *rhs, size_t size);
int rr_ucmp_impl(const void *lhs, const void *rhs, size_t size);

//
// Operator implementation macros
//

#define rr_operator_imm(type, constant)            \
  (__extension__({                                 \
    type rr_r;                                     \
    rr_##imm_impl(&rr_r, sizeof(rr_r), #constant); \
    rr_r;                                          \
  }))

#define rr_operator_cast(operator, type, x)                         \
  (__extension__({                                                  \
    __typeof__(x) rr_x = (x);                                       \
    type rr_r;                                                      \
    rr_##operator##_impl(&rr_x, sizeof(rr_x), &rr_r, sizeof(rr_r)); \
    rr_r;                                                           \
  }))

#define rr_operator_1(operator, x)             \
  (__extension__({                             \
    __typeof__(x) rr_x = (x);                  \
    rr_##operator##_impl(&rr_x, sizeof(rr_x)); \
    rr_x;                                      \
  }))

#define rr_operator_2(operator, x, y)                 \
  (__extension__({                                    \
    __typeof__(x) rr_x = (x);                         \
    __typeof__(y) rr_y = (y);                         \
    rr_##operator##_impl(&rr_x, &rr_y, sizeof(rr_x)); \
    rr_x;                                             \
  }))

#define rr_operator_inc(operator, x)           \
  (__extension__({                             \
    __typeof__(x) *rr_x = &(x);                \
    rr_##operator##_impl(rr_x, sizeof(*rr_x)); \
    *rr_x;                                     \
  }))

#define rr_operator_post_inc(operator, x)      \
  (__extension__({                             \
    __typeof__(x) *rr_x = &(x);                \
    __typeof__(*rr_x) rr_r = *rr_x;            \
    rr_##operator##_impl(rr_x, sizeof(*rr_x)); \
    rr_r;                                      \
  }))

#define rr_operator_test(x)            \
  (__extension__({                     \
    __typeof__(x) rr_x = (x);          \
    rr_test_impl(&rr_x, sizeof(rr_x)); \
  }))

#define rr_operator_cmp(operator, x, y)               \
  (__extension__({                                    \
    __typeof__(x) rr_x = (x);                         \
    __typeof__(y) rr_y = y;                           \
    rr_##operator##_impl(&rr_x, &rr_y, sizeof(rr_x)); \
  }))

//
// User-facing operator macros
//

#define rr_imm(type, x) rr_operator_imm(type, x)

#define rr_zext(type, x) rr_operator_cast(zext, type, x)
#define rr_sext(type, x) rr_operator_cast(sext, type, x)
#define rr_truncate(type, x) rr_operator_cast(truncate, type, x)

#define rr_neg(x) rr_operator_1(neg, x)
#define rr_add(x, y) rr_operator_2(add, x, y)
#define rr_sub(x, y) rr_operator_2(sub, x, y)
#define rr_mul(x, y) rr_operator_2(mul, x, y)
#define rr_udiv(x, y) rr_operator_2(udiv, x, y)
#define rr_sdiv(x, y) rr_operator_2(sdiv, x, y)
#define rr_srem(x, y) rr_operator_2(srem, x, y)
#define rr_urem(x, y) rr_operator_2(urem, x, y)

#define rr_shl(x, y) rr_operator_2(shl, x, y)
#define rr_shr(x, y) rr_operator_2(shr, x, y)
#define rr_sar(x, y) rr_operator_2(sar, x, y)

#define rr_bitnot(x) rr_operator_1(bitnot, x)
#define rr_bitand(x, y) rr_operator_2(bitand, x, y)
#define rr_bitor(x, y) rr_operator_2(bitor, x, y)
#define rr_bitxor(x, y) rr_operator_2(bitxor, x, y)

#define rr_inc(x) rr_operator_inc(inc, x)
#define rr_dec(x) rr_operator_inc(dec, x)

#define rr_post_inc(x) rr_operator_post_inc(inc, x)
#define rr_post_dec(x) rr_operator_post_inc(dec, x)

#define rr_test(x) rr_operator_test(x)

#define rr_eq(x, y) (rr_operator_cmp(ecmp, x, y) == 0)
#define rr_ne(x, y) (rr_operator_cmp(ecmp, x, y) != 0)
#define rr_slt(x, y) (rr_operator_cmp(scmp, x, y) < 0)
#define rr_ult(x, y) (rr_operator_cmp(ucmp, x, y) < 0)
#define rr_sgt(x, y) (rr_operator_cmp(scmp, x, y) > 0)
#define rr_ugt(x, y) (rr_operator_cmp(ucmp, x, y) > 0)
#define rr_sle(x, y) (rr_operator_cmp(scmp, x, y) <= 0)
#define rr_ule(x, y) (rr_operator_cmp(ucmp, x, y) <= 0)
#define rr_sge(x, y) (rr_operator_cmp(scmp, x, y) >= 0)
#define rr_uge(x, y) (rr_operator_cmp(ucmp, x, y) >= 0)
