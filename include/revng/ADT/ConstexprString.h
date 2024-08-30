#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <array>
#include <string_view>

namespace compile_time {

namespace detail {

template<std::ranges::range Range>
using DecayedElement = std::decay_t<decltype(*std::declval<Range>().begin())>;

template<std::size_t... Indices, std::ranges::range Range>
constexpr std::array<DecayedElement<Range>, sizeof...(Indices)>
makeArray(std::index_sequence<Indices...> Is, Range const &R) {
  return std::array<DecayedElement<Range>, sizeof...(Indices)>{
    { *(R.begin() + Indices)... }
  };
}

} // namespace detail

/// Makes an array out of an arbitrary constexpr range
/// (for example, a string_view).
template<std::size_t ElementCount, std::ranges::range Range>
constexpr auto makeArray(Range &&R) {
  return detail::makeArray(std::make_index_sequence<ElementCount>(),
                           std::forward<Range>(R));
}

} // namespace compile_time

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

  consteval ConstexprString(const std::array<char, N - 1> Value) noexcept :
    String(Value) {}

  consteval ConstexprString(std::ranges::range auto &&Range) noexcept :
    String(compile_time::makeArray<N - 1>(Range)) {}

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
