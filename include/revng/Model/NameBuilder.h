#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/Model/ArrayType.h"
#include "revng/Model/Binary.h"
#include "revng/Model/CABIFunctionDefinition.h"
#include "revng/Model/EnumDefinition.h"
#include "revng/Model/Function.h"
#include "revng/Model/Helpers.h"
#include "revng/Model/NamingConfiguration.h"
#include "revng/Model/RawFunctionDefinition.h"
#include "revng/Model/StructDefinition.h"
#include "revng/Model/UnionDefinition.h"
#include "revng/Support/CommonOptions.h"

namespace model {

template<typename Inheritor>
class NameBuilder {
private:
  using RFT = model::RawFunctionDefinition;

public:
  const model::NamingConfiguration &Configuration;

public:
  NameBuilder(const model::Binary &Binary) :
    Configuration(Binary.Configuration().Naming()) {}

public:
  [[nodiscard]] llvm::Error isNameReserved(llvm::StringRef Name) const {
    return static_cast<const Inheritor &>(*this).isNameReserved(Name);
  }

private:
  void assertNameIsReserved(llvm::StringRef Name) const {
    if (llvm::Error Error = isNameReserved(Name)) {
      // All good, we want an error here.
      llvm::consumeError(std::move(Error));

    } else {
      std::string ActualError = "An automatic name '" + Name.str()
                                + "' is reserved and should therefore be "
                                  "banned in order to avoid collisions.";
      revng_abort(ActualError.c_str());
    }
  }

  [[nodiscard]] std::string automaticName(const model::Binary &Binary,
                                          const model::Segment &Segment) const {
    auto Iterator = Binary.Segments().find(Segment.key());
    auto Result = std::string(Configuration.unnamedSegmentPrefix())
                  + std::to_string(std::distance(Binary.Segments().begin(),
                                                 Iterator));
    assertNameIsReserved(Result);
    return Result;
  }
  [[nodiscard]] std::string
  automaticName(const model::Function &Function) const {
    std::string Result = Configuration.unnamedFunctionPrefix().str()
                         + Function.Entry().toIdentifier();
    assertNameIsReserved(Result);
    return Result;
  }
  [[nodiscard]] std::string
  automaticName(const model::TypeDefinition &Definition) const {
    auto K = model::TypeDefinitionKind::automaticNamePrefix(Definition.Kind());
    std::string Result = Configuration.unnamedTypeDefinitionPrefix().str()
                         + K.str() + std::to_string(Definition.ID());
    assertNameIsReserved(Result);
    return Result;
  }

  [[nodiscard]] std::string
  automaticName(const model::EnumDefinition &Definition,
                const model::EnumEntry &Entry) const {
    std::string Result = Configuration.unnamedEnumEntryPrefix().str()
                         + name(Definition) + "_"
                         + std::to_string(Entry.Value());
    assertNameIsReserved(Result);
    return Result;
  }
  [[nodiscard]] std::string
  automaticName(const model::StructDefinition &Definition,
                const model::StructField &Field) const {
    std::string Result = Configuration.unnamedStructFieldPrefix().str()
                         + std::to_string(Field.Offset());
    assertNameIsReserved(Result);
    return Result;
  }
  [[nodiscard]] std::string
  automaticName(const model::UnionDefinition &Definition,
                const model::UnionField &Field) const {
    std::string Result = Configuration.unnamedUnionFieldPrefix().str()
                         + std::to_string(Field.Index());
    assertNameIsReserved(Result);
    return Result;
  }

  [[nodiscard]] std::string
  automaticName(const model::CABIFunctionDefinition &Function,
                const model::Argument &Argument) const {
    std::string Result = Configuration.unnamedFunctionArgumentPrefix().str()
                         + std::to_string(Argument.Index());
    assertNameIsReserved(Result);
    return Result;
  }
  [[nodiscard]] std::string
  automaticName(const model::RawFunctionDefinition &Function,
                const model::NamedTypedRegister &Argument) const {
    std::string Result = Configuration.unnamedFunctionRegisterPrefix().str()
                         + std::string(getRegisterName(Argument.Location()));
    assertNameIsReserved(Result);
    return Result;
  }

public:
  /// These method (and its overloads) should be the only way you obtain any
  /// name you embed into an artifact (be it decompiled c, disassembly, or
  /// anything else).
  ///
  /// That ensures that the names always comply with our policy for names.
  /// Which is as follows:
  ///
  /// Users can give _any_ names to the objects they are allowed to rename
  /// in the model and as long as such names are unique, the model will be
  /// valid (*).
  /// BUT, such names can sometimes be suppressed and replaced by their
  /// corresponding automatic names (every user-renamable object always has
  /// one). That happens if:
  /// 1. The name contains symbols not allowed to be emitted by underlying
  ///    language, for example anything outside of `[A-Za-z0-9_]` for C.
  /// 2. It collides with something underlying language reserves, for example
  ///    we will not emit a function named `if` in C, or `rax` in assembly.
  /// 3. It collides with something we reserve, which includes:
  ///      1. automatic names for every renamable object, for example, struct
  ///         names: `struct_42`.
  ///      2. names for static stuff that cannot be renamed, for example,
  ///         primitive names: `uint64_t`.
  ///      3. names for dynamic stuff that cannot be renamed, for example,
  ///         helper names: `page_dump`.
  ///      4. some extra prefixes, like everything starting with `__builtin_`.
  ///
  /// ------------------------
  /// (*) Temporarily, there are further restrictions, namely you are not
  /// allowed to use `/`. But these will eventually be lifted.
  [[nodiscard]] std::string name(EntityWithName auto const &E) const {
    if (E.Name().empty()) {
      return automaticName(E);

    } else if (llvm::Error Error = isNameReserved(E.Name())) {
      // We don't care what the specific error is - if there is one,
      // just fall back on the automatic name.
      llvm::consumeError(std::move(Error));

      return automaticName(E);

    } else {
      return E.Name();
    }
  }

  [[nodiscard]] std::string name(const auto &Parent,
                                 EntityWithName auto const &E) const {
    if (E.Name().empty()) {
      return automaticName(Parent, E);

    } else if (llvm::Error Error = isNameReserved(E.Name())) {
      // We don't care what the specific error is - if there is one,
      // just fall back on the automatic name.
      llvm::consumeError(std::move(Error));

      return automaticName(Parent, E);

    } else {
      return E.Name();
    }
  }

  // Dynamic functions are special - we never introduce automatic names for them
  [[nodiscard]] std::string name(const model::DynamicFunction &Function) const {
    llvm::Error Error = isNameReserved(Function.Name());
    if (Error) {
      std::string ErrorMessage = "Dynamic function name `" + Function.Name()
                                 + "` is not allowed: "
                                 + revng::unwrapError(std::move(Error));
      revng_abort(ErrorMessage.c_str());
    }

    return Function.Name();
  }

  [[nodiscard]] std::string llvmName(const model::Function &Function) const {
    if (DebugNames)
      return "local_" + name(Function);
    else
      return "local_" + Function.Entry().toString();
  }

private:
  [[nodiscard]] std::optional<std::string>
  warningImpl(const std::string &Name,
              const std::string &OriginalName,
              llvm::Error &&Reason) const {
    if (Name == OriginalName) {
      // Suppress warnings on appropriately placed automatic names in the model.
      //
      // TODO: get rid of this once `import-from-c` is fixed and we no longer
      //       put such names in.
      llvm::consumeError(std::move(Reason));
      return std::nullopt;
    }

    return "Name '" + OriginalName
           + "' is not valid, so it was replaced by an automatic one ('" + Name
           + "') because " + revng::unwrapError(std::move(Reason)) + ".";
  }

public:
  [[nodiscard]] std::optional<std::string>
  warning(EntityWithName auto const &E) const {
    if (E.Name().empty())
      return std::nullopt;

    if (auto Reason = isNameReserved(E.Name()))
      return warningImpl(automaticName(E), E.Name(), std::move(Reason));

    return std::nullopt;
  }

  [[nodiscard]] std::optional<std::string>
  warning(const auto &Parent, EntityWithName auto const &E) const {
    if (E.Name().empty())
      return std::nullopt;

    if (auto Reason = isNameReserved(E.Name()))
      return warningImpl(automaticName(Parent, E), E.Name(), std::move(Reason));

    return std::nullopt;
  }

public:
  [[nodiscard]] std::string paddingFieldName(uint64_t Offset) const {
    std::string Result = Configuration.structPaddingPrefix().str()
                         + std::to_string(Offset);
    assertNameIsReserved(Result);
    return Result;
  }

public:
  [[nodiscard]] std::string
  artificialReturnValueWrapperName(const RFT &Function) const {
    auto Result = Configuration.artificialReturnValuePrefix().str()
                  + name(Function);
    assertNameIsReserved(Result);
    return Result;
  }

private:
  RecursiveCoroutine<std::string>
  artificialArrayWrapperNameImpl(const model::Type &Type) const {
    if (auto *Array = llvm::dyn_cast<model::ArrayType>(&Type)) {
      std::string Result = "array_" + std::to_string(Array->ElementCount())
                           + "_of_";
      Result += rc_recur artificialArrayWrapperNameImpl(*Array->ElementType());
      rc_return Result;

    } else if (auto *D = llvm::dyn_cast<model::DefinedType>(&Type)) {
      std::string Result = (D->IsConst() ? "const_" : "");
      rc_return Result += this->name(D->unwrap());

    } else if (auto *Ptr = llvm::dyn_cast<model::PointerType>(&Type)) {
      std::string Result = (D->IsConst() ? "const_ptr_to_" : "ptr_to_");
      Result += rc_recur artificialArrayWrapperNameImpl(*Ptr->PointeeType());
      rc_return Result;

    } else if (auto *Primitive = llvm::dyn_cast<model::PrimitiveType>(&Type)) {
      std::string Result = (D->IsConst() ? "const_" : "");
      rc_return Result += Primitive->getCName();

    } else {
      revng_abort("Unsupported model::Type.");
    }
  }

public:
  [[nodiscard]] std::string
  artificialArrayWrapperName(const model::ArrayType &Type) const {
    auto Result = Configuration.artificialArrayWrapperPrefix().str()
                  + std::string(artificialArrayWrapperNameImpl(Type));
    assertNameIsReserved(Result);
    return Result;
  }

  [[nodiscard]] std::string artificialArrayWrapperFieldName() const {
    auto Result = Configuration.artificialArrayWrapperFieldName().str();
    assertNameIsReserved(Result);
    return Result;
  }
};

struct CNameBuilder : public NameBuilder<CNameBuilder> {
public:
  using NameBuilder<CNameBuilder>::NameBuilder;

  [[nodiscard]] llvm::Error isNameReserved(llvm::StringRef Name) const;
};
struct AssemblyNameBuilder : NameBuilder<AssemblyNameBuilder> {
public:
  const model::Architecture::Values &Architecture;

public:
  AssemblyNameBuilder(const model::Binary &Binary) :
    NameBuilder<AssemblyNameBuilder>(Binary),
    Architecture(Binary.Architecture()) {}

  [[nodiscard]] llvm::Error isNameReserved(llvm::StringRef Name) const;
};

inline std::string sanitizeHelperName(llvm::StringRef Name) {
  auto Replace = std::views::transform([](char Character) -> char {
    return std::isalnum(Character) ? Character : '_';
  });
  return Name | Replace | revng::to<std::string>();
}

} // namespace model
