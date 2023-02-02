#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstddef>
#include <tuple>
#include <utility>

#include "llvm/ADT/StringRef.h"

namespace pipeline {
template<typename T>
class Option {
public:
  constexpr Option(const char *Name, T Default) :
    Default(std::forward<T>(Default)), Name(Name) {}

  using Type = std::decay_t<T>;
  T Default;
  const char *Name;
};

template<typename T>
concept HasOptions = requires(T) {
  { T::Options };
};

/// TODO: Remove after updating to clang-format with concept support.
struct ClangFormatPleaseDoNotBreakMyCode;
// clang-format off
// clang-format on

namespace detail {

template<typename Invokable, size_t Index>
constexpr auto &getOptionInfo() {
  return std::get<Index>(Invokable::Options);
}

template<typename Invokable, size_t Index>
llvm::StringRef getOptionName() {
  return getOptionInfo<Invokable, Index>().Name;
}

template<typename T, size_t Index>
using ArgTypeImpl = std::decay_t<decltype(getOptionInfo<T, Index>())>;

template<typename InvokableType, size_t Index>
using OptionTypeImpl = typename ArgTypeImpl<InvokableType, Index>::Type;

template<typename InvokableType, size_t Index>
constexpr bool
  IsConstCharPtr = std::is_same_v<OptionTypeImpl<InvokableType, Index>,
                                  const char *>;

template<typename InvokableType, size_t Index>
using OptionType = std::conditional_t<IsConstCharPtr<InvokableType, Index>,
                                      std::string,
                                      OptionTypeImpl<InvokableType, Index>>;

template<typename Invokable, size_t Index>
llvm::StringRef getTypeName() {
  return typeNameImpl<ArgTypeImpl<Invokable, Index>>();
}

template<typename InvokableType, size_t Index>
OptionType<InvokableType, Index> getOptionDefault() {
  return getOptionInfo<InvokableType, Index>().Default;
}
} // namespace detail
} // namespace pipeline
