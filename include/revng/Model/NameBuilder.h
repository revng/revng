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
#include "revng/Model/Identifier.h"
#include "revng/Model/NamingConfiguration.h"
#include "revng/Model/RawFunctionDefinition.h"
#include "revng/Model/StructDefinition.h"
#include "revng/Model/UnionDefinition.h"
#include "revng/Support/CommonOptions.h"

namespace model {

class NameBuilder {
private:
  using RFT = model::RawFunctionDefinition;

public:
  const model::NamingConfiguration &Configuration;

public:
  NameBuilder(const model::Binary &Binary) :
    Configuration(Binary.Configuration().Naming()) {}

public:
  [[nodiscard]] static bool
  isNameReserved(llvm::StringRef Name,
                 const model::NamingConfiguration &Configuration);

  [[nodiscard]] llvm::Error isNameReserved(llvm::StringRef Name) const {
    if (isNameReserved(Name, Configuration))
      return llvm::Error::success();
    else
      return revng::createError("Name is reserved: " + Name);
  }

private:
  static constexpr bool EnableSanityChecks = true;
  static void failSanityCheck(llvm::StringRef Name) {
    std::string Error = "An automatic name '" + Name.str()
                        + "' must not be allowed, otherwise a name collision "
                          "might occur.";
    revng_abort(Error.c_str());
  }

  [[nodiscard]] std::string automaticName(const model::Segment &Segment) const {
    std::string Result = Configuration.unnamedSegmentPrefix().str()
                         + Segment.StartAddress().toIdentifier() + "_"
                         + std::to_string(Segment.VirtualSize());
    if constexpr (EnableSanityChecks)
      if (not isNameReserved(Result, Configuration))
        failSanityCheck(Result);
    return Result;
  }
  [[nodiscard]] std::string
  automaticName(const model::Function &Function) const {
    std::string Result = Configuration.unnamedFunctionPrefix().str()
                         + Function.Entry().toIdentifier();
    if constexpr (EnableSanityChecks)
      if (not isNameReserved(Result, Configuration))
        failSanityCheck(Result);
    return Result;
  }
  [[nodiscard]] std::string
  automaticName(const model::TypeDefinition &Definition) const {
    auto K = model::TypeDefinitionKind::automaticNamePrefix(Definition.Kind());
    std::string Result = Configuration.unnamedTypeDefinitionPrefix().str()
                         + K.str() + std::to_string(Definition.ID());
    if constexpr (EnableSanityChecks)
      if (not isNameReserved(Result, Configuration))
        failSanityCheck(Result);
    return Result;
  }

  [[nodiscard]] std::string
  automaticName(const model::EnumDefinition &Definition,
                const model::EnumEntry &Entry) {
    std::string Result = Configuration.unnamedEnumEntryPrefix().str()
                         + name(Definition) + "_"
                         + std::to_string(Entry.Value());
    if constexpr (EnableSanityChecks)
      if (not isNameReserved(Result, Configuration))
        failSanityCheck(Result);
    return Result;
  }
  [[nodiscard]] std::string
  automaticName(const model::StructDefinition &Definition,
                const model::StructField &Field) const {
    std::string Result = Configuration.unnamedStructFieldPrefix().str()
                         + std::to_string(Field.Offset());
    if constexpr (EnableSanityChecks)
      if (not isNameReserved(Result, Configuration))
        failSanityCheck(Result);
    return Result;
  }
  [[nodiscard]] std::string
  automaticName(const model::UnionDefinition &Definition,
                const model::UnionField &Field) const {
    std::string Result = Configuration.unnamedUnionFieldPrefix().str()
                         + std::to_string(Field.Index());
    if constexpr (EnableSanityChecks)
      if (not isNameReserved(Result, Configuration))
        failSanityCheck(Result);
    return Result;
  }

  [[nodiscard]] std::string
  automaticName(const model::CABIFunctionDefinition &Function,
                const model::Argument &Argument) const {
    std::string Result = Configuration.unnamedFunctionArgumentPrefix().str()
                         + std::to_string(Argument.Index());
    if constexpr (EnableSanityChecks)
      if (not isNameReserved(Result, Configuration))
        failSanityCheck(Result);
    return Result;
  }
  [[nodiscard]] std::string
  automaticName(const model::RawFunctionDefinition &Function,
                const model::NamedTypedRegister &Argument) const {
    std::string Result = Configuration.unnamedFunctionRegisterPrefix().str()
                         + std::string(getRegisterName(Argument.Location()));
    if constexpr (EnableSanityChecks)
      if (not isNameReserved(Result, Configuration))
        failSanityCheck(Result);
    return Result;
  }

public:
  [[nodiscard]] std::string name(EntityWithCustomName auto const &Entity) {
    if (Entity.CustomName().empty()
        || isNameReserved(Entity.CustomName(), Configuration))
      return automaticName(Entity);

    return Entity.CustomName().str().str();
  }

  [[nodiscard]] std::string name(const auto &Parent,
                                 EntityWithCustomName auto const &Entity) {
    if (Entity.CustomName().empty()
        || isNameReserved(Entity.CustomName(), Configuration))
      return automaticName(Parent, Entity);

    return Entity.CustomName().str().str();
  }

  // Dynamic functions are special - we never introduce automatic names for them
  [[nodiscard]] std::string name(const model::DynamicFunction &Function) {
    revng_assert(not isNameReserved(Function.CustomName(), Configuration));
    return Function.CustomName().str().str();
  }

  [[nodiscard]] std::string llvmName(const model::Function &Function) {
    if (DebugNames)
      return "local_" + name(Function);
    else
      return "local_" + Function.Entry().toString();
  }

public:
  [[nodiscard]] std::string paddingFieldName(uint64_t Offset) const {
    std::string Result = Configuration.structPaddingPrefix().str()
                         + std::to_string(Offset);
    if constexpr (EnableSanityChecks)
      if (not isNameReserved(Result, Configuration))
        failSanityCheck(Result);
    return Result;
  }

public:
  [[nodiscard]] std::string
  artificialReturnValueWrapperName(const RFT &Function) {
    auto Result = Configuration.artificialReturnValuePrefix().str()
                  + name(Function);
    if constexpr (EnableSanityChecks)
      if (not isNameReserved(Result, Configuration))
        failSanityCheck(Result);
    return Result;
  }

private:
  RecursiveCoroutine<std::string>
  artificialArrayWrapperNameImpl(const model::Type &Type);

public:
  [[nodiscard]] std::string
  artificialArrayWrapperName(const model::ArrayType &Type) {
    auto Result = Configuration.artificialArrayWrapperPrefix().str()
                  + std::string(artificialArrayWrapperNameImpl(Type));
    if constexpr (EnableSanityChecks)
      if (not isNameReserved(Result, Configuration))
        failSanityCheck(Result);
    return Result;
  }

  [[nodiscard]] std::string artificialArrayWrapperFieldName() {
    auto Result = Configuration.artificialArrayWrapperFieldName().str();
    if constexpr (EnableSanityChecks)
      if (not isNameReserved(Result, Configuration))
        failSanityCheck(Result);
    return Result;
  }
};

} // namespace model
