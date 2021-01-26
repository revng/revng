#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/YAMLTraits.h"

namespace detail {

using namespace llvm::yaml;

template<typename T>
using has_MappingTraits = has_MappingTraits<T, EmptyContext>;

template<typename T, typename K = void>
using ei_hmt = std::enable_if_t<has_MappingTraits<T>::value, K>;
} // namespace detail

template<typename T, typename K = void>
using enable_if_has_MappingTraits = detail::ei_hmt<T, K>;

template<typename T>
inline llvm::StringRef getNameFromYAMLScalar(T V) {
  struct GetScalarIO {
    llvm::StringRef Result;
    void enumCase(const T &V, llvm::StringRef Name, const T &M) {
      if (V == M) {
        Result = Name;
      }
    }
  };
  GetScalarIO ExtractName;
  llvm::yaml::ScalarEnumerationTraits<T>::enumeration(ExtractName, V);

  return ExtractName.Result;
}

template<typename T>
inline T getValueFromYAMLScalar(llvm::StringRef Name) {
  struct GetScalarIO {
    llvm::StringRef TargetName;
    void enumCase(T &V, llvm::StringRef Name, const T &M) {
      if (TargetName == Name)
        V = M;
    }
  };
  T Result;
  GetScalarIO ExtractValue{ Name };
  llvm::yaml::ScalarEnumerationTraits<T>::enumeration(ExtractValue, Result);

  return Result;
}
