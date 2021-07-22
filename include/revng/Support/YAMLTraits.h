#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/YAMLTraits.h"

#include "revng/Support/Assert.h"

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

template<typename T, char Separator>
struct CompositeScalar {
  static_assert(std::tuple_size_v<T> >= 0);

  template<size_t I = 0>
  static void output(const T &Value, void *Ctx, llvm::raw_ostream &Output) {
    if constexpr (I < std::tuple_size_v<T>) {

      if constexpr (I != 0) {
        Output << Separator;
      }

      using element = std::tuple_element_t<I, T>;
      Output << getNameFromYAMLScalar<element>(get<I>(Value));

      CompositeScalar::output<I + 1>(Value, Ctx, Output);
    }
  }

  template<size_t I = 0>
  static llvm::StringRef input(llvm::StringRef Scalar, void *Ctx, T &Value) {
    if constexpr (I < std::tuple_size_v<T>) {
      auto [Before, After] = Scalar.split(Separator);

      using element = std::tuple_element_t<I, T>;
      get<I>(Value) = getValueFromYAMLScalar<element>(Before);

      return CompositeScalar::input<I + 1>(After, Ctx, Value);
    } else {
      revng_assert(Scalar.size() == 0);
      return Scalar;
    }
  }

  static llvm::yaml::QuotingType mustQuote(llvm::StringRef) {
    return llvm::yaml::QuotingType::Double;
  }
};
