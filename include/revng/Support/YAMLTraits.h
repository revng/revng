#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include <bit>

#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/ADT/KeyedObjectContainer.h"
#include "revng/Support/Assert.h"
#include "revng/TupleTree/TupleTreeCompatible.h"

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

namespace revng::detail {
using EC = llvm::yaml::EmptyContext;
template<typename T>
concept MappableWithEmptyContext = llvm::yaml::has_MappingTraits<T, EC>::value;
} // namespace revng::detail

template<typename T>
concept Yamlizable = llvm::yaml::has_DocumentListTraits<T>::value
                     or revng::detail::MappableWithEmptyContext<T>
                     or llvm::yaml::has_SequenceTraits<T>::value
                     or llvm::yaml::has_BlockScalarTraits<T>::value
                     or llvm::yaml::has_CustomMappingTraits<T>::value
                     or llvm::yaml::has_PolymorphicTraits<T>::value
                     or llvm::yaml::has_ScalarTraits<T>::value
                     or llvm::yaml::has_ScalarEnumerationTraits<T>::value;

namespace revng::detail {

struct NoYaml {};

static_assert(not Yamlizable<NoYaml>);

} // end namespace revng::detail

static_assert(Yamlizable<int>);
static_assert(Yamlizable<std::vector<int>>);

constexpr inline auto IsYamlizable = [](auto *K) {
  return Yamlizable<std::remove_pointer_t<decltype(K)>>;
};

// How to improve performance without losing safety of a `TupleTree`:
//
// * `TupleTreeReference` must contain a `std::variant` between what they
//   have right now and a naked pointer.
// * The `operator* const` of `UpcastablePointer` (which should be
//   renamed to *Variant*) should return a constant reference. Same
//   for `TupleTreeReference`.
// * `TupleTree` should have:
//   * `const TupleTree freeze()`: `std::move` itself in the `const`
//     result and transforms all the `TupleTreeReference`s in direct
//     pointers.
//   * `TupleTree unfreeze()`: `std::move` itself in the `const`
//     result and transforms all the `TupleTreeReference`s in root +
//     key.
// * Alternatively, we could push the functionality of `ModelWrapper`
//   into `TupleTree`. In this way, the default behavior would be to
//   be frozen. A RAII wrapper could take care of unfreeze and
//   refreeze the TupleTree.

// TODO: `const` stuff is not YAML-serializable
template<typename S, Yamlizable T>
void serialize(S &Stream, T &Element) {
  if constexpr (std::is_base_of_v<llvm::raw_ostream, S>) {
    if constexpr (HasScalarOrEnumTraits<T>) {
      Stream << llvm::StringRef(getNameFromYAMLScalar(Element));
    } else {
      llvm::yaml::Output YAMLOutput(Stream);
      YAMLOutput << Element;
    }
  } else {
    std::string Buffer;
    if constexpr (HasScalarOrEnumTraits<T>) {
      Buffer = getNameFromYAMLScalar(Element);
    } else {
      llvm::raw_string_ostream StringStream(Buffer);
      llvm::yaml::Output YAMLOutput(StringStream);
      YAMLOutput << Element;
    }
    Stream << Buffer;
  }
}

template<typename S, Yamlizable T>
void serialize(S &Stream, const T &Element) {
  serialize(Stream, const_cast<T &>(Element));
}

template<Yamlizable T>
llvm::Error serializeToFile(const T &ToWrite, const llvm::StringRef &Path) {
  std::error_code ErrorCode;
  llvm::raw_fd_ostream OutFile(Path, ErrorCode, llvm::sys::fs::CD_CreateAlways);
  if (!!ErrorCode) {
    return llvm::make_error<llvm::StringError>("Could not open file "
                                                 + Path.str(),
                                               ErrorCode);
  }

  serialize(OutFile, ToWrite);

  return llvm::Error::success();
}

template<Yamlizable T>
std::string serializeToString(const T &ToDump) {
  std::string Buffer;
  {
    llvm::raw_string_ostream StringStream(Buffer);
    serialize(StringStream, ToDump);
  }
  return Buffer;
}

namespace revng::detail {
template<typename T>
llvm::Expected<T>
deserializeImpl(llvm::StringRef YAMLString, void *Context = nullptr) {
  if constexpr (HasScalarOrEnumTraits<T>) {
    return getValueFromYAMLScalar<T>(YAMLString);
  } else {
    T Result;

    llvm::yaml::Input YAMLInput(YAMLString, Context);
    YAMLInput >> Result;

    std::error_code EC = YAMLInput.error();
    if (EC)
      return llvm::errorCodeToError(EC);

    return Result;
  }
}

} // namespace revng::detail

template<NotTupleTreeCompatible T>
llvm::Expected<T>
deserialize(llvm::StringRef YAMLString, void *Context = nullptr) {
  return revng::detail::deserializeImpl<T>(YAMLString, Context);
}

template<NotTupleTreeCompatible T>
llvm::Expected<T>
deserializeFileOrSTDIN(const llvm::StringRef &Path, void *Context = nullptr) {
  auto MaybeBuffer = llvm::MemoryBuffer::getFileOrSTDIN(Path);
  if (not MaybeBuffer)
    return llvm::errorCodeToError(MaybeBuffer.getError());

  return deserialize<T>((*MaybeBuffer)->getBuffer(), Context);
}

template<HasScalarTraits T>
struct llvm::yaml::ScalarTraits<std::tuple<T>> {
  using ValueType = std::tuple<T>;
  using ValueTrait = llvm::yaml::ScalarTraits<T>;

  static void
  output(const ValueType &Value, void *Ctx, llvm::raw_ostream &Output) {
    ValueTrait().output(std::get<0>(Value), Ctx, Output);
  }

  static llvm::StringRef
  input(llvm::StringRef Scalar, void *Ctx, ValueType &Value) {
    return ValueTrait().input(Scalar, Ctx, std::get<0>(Value));
  }

  static llvm::yaml::QuotingType mustQuote(llvm::StringRef String) {
    return ValueTrait().mustQuote(String);
  }
};

template<>
struct llvm::yaml::ScalarTraits<std::byte> {
  static_assert(HasScalarTraits<uint8_t>);
  static_assert(sizeof(std::byte) == sizeof(uint8_t));

  static void output(const std::byte &Value, void *, llvm::raw_ostream &Out) {
    Out << std::bit_cast<uint8_t>(Value);
  }

  static StringRef input(StringRef Scalar, void *Ptr, std::byte &Value) {
    uint8_t Temporary;
    auto Err = llvm::yaml::ScalarTraits<uint8_t>::input(Scalar, Ptr, Temporary);
    if (Err.empty())
      Value = std::bit_cast<std::byte>(Temporary);
    return Err;
  }

  static QuotingType mustQuote(StringRef Scalar) {
    return llvm::yaml::ScalarTraits<uint8_t>::mustQuote(Scalar);
  }
};
LLVM_YAML_IS_FLOW_SEQUENCE_VECTOR(std::byte);
