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
inline llvm::StringRef getNameFromYAMLEnumScalar(T V) {
  using namespace llvm::yaml;
  static_assert(has_ScalarEnumerationTraits<T>::value);
  struct GetScalarIO {
    llvm::StringRef Result;
    void enumCase(const T &V,
                  llvm::StringRef Name,
                  const T &M,
                  llvm::yaml::QuotingType = llvm::yaml::QuotingType::None) {
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
inline std::string getNameFromYAMLScalar(T V) {
  using namespace llvm::yaml;
  static_assert(has_ScalarTraits<T>::value
                or has_ScalarEnumerationTraits<T>::value);

  if constexpr (has_ScalarTraits<T>::value) {
    std::string Buffer;
    llvm::raw_string_ostream Stream(Buffer);
    llvm::yaml::ScalarTraits<T>::output(V, nullptr, Stream);
    return Buffer;
  } else {
    return getNameFromYAMLEnumScalar(V);
  }
}

template<typename T>
T getInvalidValueFromYAMLScalar() {
  // Default action: abort. Users can override this behavior.
  revng_abort();
}

template<typename T>
inline T getValueFromYAMLScalar(llvm::StringRef Name) {
  using namespace llvm::yaml;
  static_assert(has_ScalarTraits<T>::value
                or has_ScalarEnumerationTraits<T>::value);

  T Result;

  if constexpr (has_ScalarTraits<T>::value) {
    llvm::yaml::ScalarTraits<T>::input(Name, nullptr, Result);
  } else {
    struct GetScalarIO {
      bool Found = false;
      llvm::StringRef TargetName;
      void enumCase(T &V,
                    llvm::StringRef Name,
                    const T &M,
                    llvm::yaml::QuotingType = llvm::yaml::QuotingType::None) {
        if (TargetName == Name) {
          revng_assert(not Found);
          Found = true;
          V = M;
        }
      }
    };
    GetScalarIO ExtractValue{ false, Name };
    llvm::yaml::ScalarEnumerationTraits<T>::enumeration(ExtractValue, Result);
    if (not ExtractValue.Found)
      Result = getInvalidValueFromYAMLScalar<T>();
  }

  return Result;
}
