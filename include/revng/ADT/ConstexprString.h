#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <array>
#include <string_view>

/// Allows passing strings as template parameters
///
/// Objects of this type should only ever be constructed from a string literal.
/// Internally, it stores the string (minus the last '\0' character as an array)
template<size_t N = 1>
struct ConstexprString {
public:
  /// The data members have to be public for a literal type to be usable as
  /// a template parameter :{
  ///
  /// That being said, please, think of it as a private field and never access
  /// it directly.
  std::array<char, N - 1> String;

  constexpr ConstexprString(const char (&Value)[N]) noexcept
    requires(N >= 2)
  {
    std::copy(Value, Value + N - 1, String.data());
  }

  constexpr ConstexprString() noexcept
    requires(N == 1)
    : String({}) {}

  constexpr ConstexprString(const ConstexprString &) = default;
  constexpr ConstexprString(ConstexprString &&) = default;
  constexpr ConstexprString &operator=(const ConstexprString &) = default;
  constexpr ConstexprString &operator=(ConstexprString &&) = default;

  constexpr size_t size() const noexcept { return String.size(); }
  constexpr const char *begin() const noexcept { return String.begin(); }
  constexpr const char *end() const noexcept { return String.end(); }
  constexpr char operator[](size_t Idx) const noexcept { return String[Idx]; }

  constexpr std::string_view operator*() const noexcept {
    return operator std::string_view();
  }
  constexpr operator std::string_view() const noexcept {
    return std::string_view{ String.data(), String.size() };
  }
};
