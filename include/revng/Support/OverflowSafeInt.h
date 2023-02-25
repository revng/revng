#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <optional>

#include "llvm/Support/MathExtras.h"

#include "revng/ADT/Concepts.h"

template<std::integral T>
class OverflowSafeInt {
private:
  std::optional<T> Value;

public:
  OverflowSafeInt() {}
  OverflowSafeInt(T Value) : Value(Value) {}
  OverflowSafeInt(std::optional<T> Value) : Value(Value) {}

public:
  std::optional<T> value() const { return Value; }

  explicit operator bool() const { return Value.has_value(); }

  T operator*() {
    revng_assert(Value);
    return *Value;
  }

public:
  OverflowSafeInt operator+(const OverflowSafeInt &Other) const {
    if (not Value or not Other.Value)
      return OverflowSafeInt();
    else
      return OverflowSafeInt(add(*Value, *Other.Value));
  }

  OverflowSafeInt operator-(const OverflowSafeInt &Other) const {
    if (not Value or not Other.Value)
      return OverflowSafeInt();
    else
      return OverflowSafeInt(sub(*Value, *Other.Value));
  }

  OverflowSafeInt operator*(const OverflowSafeInt &Other) const {
    if (not Value or not Other.Value)
      return OverflowSafeInt();
    else
      return OverflowSafeInt(mul(*Value, *Other.Value));
  }

  OverflowSafeInt operator/(T Other) const {
    revng_assert(llvm::isPowerOf2_64(Other));

    if (not Value)
      return OverflowSafeInt();
    else
      return OverflowSafeInt(*Value / Other);
  }

public:
  OverflowSafeInt &operator+=(const OverflowSafeInt &Other) {
    *this = *this + Other;
    return *this;
  }

  OverflowSafeInt &operator-=(const OverflowSafeInt &Other) {
    *this = *this - Other;
    return *this;
  }

  OverflowSafeInt &operator*=(const OverflowSafeInt &Other) {
    *this = *this * Other;
    return *this;
  }

  OverflowSafeInt &operator/=(T Other) const {
    *this = *this / Other;
    return *this;
  }

private:
  static std::optional<T> add(T FirstOperand, T SecondOperand) {
    T Result = 0;
    bool Overflow = __builtin_add_overflow(FirstOperand,
                                           SecondOperand,
                                           &Result);
    if (Overflow)
      return std::nullopt;
    else
      return Result;
  }

  static std::optional<T> sub(T FirstOperand, T SecondOperand) {
    T Result = 0;
    bool Overflow = __builtin_sub_overflow(FirstOperand,
                                           SecondOperand,
                                           &Result);
    if (Overflow)
      return std::nullopt;
    else
      return Result;
  }

  static std::optional<T> mul(T FirstOperand, T SecondOperand) {
    T Result = 0;
    bool Overflow = __builtin_mul_overflow(FirstOperand,
                                           SecondOperand,
                                           &Result);
    if (Overflow)
      return std::nullopt;
    else
      return Result;
  }
};
