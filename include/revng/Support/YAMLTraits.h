#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/YAMLTraits.h"

template<typename T>
concept HasScalarTraits = llvm::yaml::has_ScalarTraits<T>::value;

template<typename T>
concept HasScalarEnumTraits = llvm::yaml::has_ScalarEnumerationTraits<T>::value;

template<typename T>
concept HasScalarOrEnumTraits = HasScalarTraits<T> or HasScalarEnumTraits<T>;

template<HasScalarEnumTraits T>
inline llvm::StringRef getNameFromYAMLEnumScalar(T V) {
  using namespace llvm::yaml;
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

template<HasScalarOrEnumTraits T>
inline std::string getNameFromYAMLScalar(T V) {
  using namespace llvm::yaml;

  if constexpr (has_ScalarTraits<T>::value) {
    std::string Buffer;
    llvm::raw_string_ostream Stream(Buffer);
    llvm::yaml::ScalarTraits<T>::output(V, nullptr, Stream);
    return Buffer;
  } else {
    return getNameFromYAMLEnumScalar(V).str();
  }
}

template<typename T>
T getInvalidValueFromYAMLScalar() {
  // Default action: abort. Users can override this behavior.
  revng_abort();
}

template<HasScalarOrEnumTraits T>
inline T getValueFromYAMLScalar(llvm::StringRef Name) {
  using namespace llvm::yaml;

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
