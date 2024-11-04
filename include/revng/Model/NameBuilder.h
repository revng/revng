#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/Model/ArrayType.h"
#include "revng/Model/CABIFunctionDefinition.h"
#include "revng/Model/EnumDefinition.h"
#include "revng/Model/Identifier.h"
#include "revng/Model/NamingConfiguration.h"
#include "revng/Model/RawFunctionDefinition.h"
#include "revng/Model/StructDefinition.h"
#include "revng/Model/UnionDefinition.h"
#include "revng/Support/CommonOptions.h"

namespace model {

namespace detail {

struct TransparentStringHash {
  using is_transparent = void;

  std::size_t operator()(const char *Value) const {
    return std::hash<std::string_view>{}(Value);
  }
  std::size_t operator()(const std::string &Value) const {
    return std::hash<std::string_view>{}(Value);
  }
  std::size_t operator()(std::string_view Value) const {
    return std::hash<std::string_view>{}(Value);
  }
  std::size_t operator()(llvm::StringRef Value) const {
    return std::hash<std::string_view>{}(Value);
  }
  std::size_t operator()(const model::Identifier &Value) const {
    return std::hash<std::string_view>{}(Value.str());
  }
};

} // namespace detail

class Binary;
class VerifyHelper;

class NameBuilder {
private:
  using GlobalNamespaceMap = std::unordered_map<std::string,
                                                std::string,
                                                detail::TransparentStringHash,
                                                std::equal_to<>>;

  std::optional<GlobalNamespaceMap> GlobalNamespace = std::nullopt;

public:
  const Binary &Binary;

private:
  using RFT = model::RawFunctionDefinition;

public:
  NameBuilder(const model::Binary &Binary) : Binary(Binary) {}

private:
  llvm::Error isNameForbidden(std::string_view Name);

public:
  const model::NamingConfiguration &configuration() const;

public:
  // This function is triggered automatically the first time a name is
  // requested.
  //
  // There's no need to call it manually unless you expect it to fail,
  // for example, when verifying a binary.
  llvm::Error populateGlobalNamespace();

  bool isGlobalNamespacePopulated() const {
    return GlobalNamespace.has_value();
  }

private:
  const GlobalNamespaceMap &globalNamespace() {
    if (not GlobalNamespace.has_value()) {
      auto MaybeError = populateGlobalNamespace();
      revng_check(not MaybeError);
      revng_assert(GlobalNamespace.has_value());
    }

    return *GlobalNamespace;
  }

public:
  [[nodiscard]] bool isGlobalSymbol(std::string_view Name) {
    return globalNamespace().contains(Name);
  }

public:
  Identifier name(const model::Segment &Segment);
  Identifier name(const model::Function &Function);
  Identifier name(const model::DynamicFunction &Function);

  std::string llvmName(const model::Function &Function) {
    std::string Result = "local_";
    if (DebugNames)
      Result += name(Function).str().str();
    else
      Result += Function.Entry().toString();

    return Result;
  }

public:
  Identifier name(const model::TypeDefinition &Definition);

  Identifier name(const model::EnumDefinition &Definition,
                  const model::EnumEntry &Entry);
  Identifier name(const model::EnumDefinition &Definition, uint64_t Value) {
    return name(Definition, Definition.Entries().at(Value));
  }

  Identifier name(const model::StructDefinition &Definition,
                  const model::StructField &Field);
  Identifier name(const model::StructDefinition &Definition, uint64_t Offset) {
    revng_assert(Offset <= Definition.Size());
    return name(Definition, Definition.Fields().at(Offset));
  }

  Identifier name(const model::UnionDefinition &Definition,
                  const model::UnionField &Field);
  Identifier name(const model::UnionDefinition &Definition, uint64_t Index) {
    revng_assert(Index <= Definition.Fields().size());
    return name(Definition, Definition.Fields().at(Index));
  }

  Identifier paddingFieldName(uint64_t Offset) const {
    // Not checking anything here since the relevant checks should have already
    // been done when constructing the helper.
    return Identifier(std::string(configuration().structPaddingPrefix())
                      + std::to_string(Offset));
  }

public:
  Identifier argumentName(const model::CABIFunctionDefinition &Function,
                          const model::Argument &Argument);
  Identifier argumentName(const model::CABIFunctionDefinition &Function,
                          uint64_t ArgumentIndex) {
    revng_assert(ArgumentIndex <= Function.Arguments().size());
    return argumentName(Function, Function.Arguments().at(ArgumentIndex));
  }

  Identifier argumentName(const model::RawFunctionDefinition &Function,
                          const model::NamedTypedRegister &Argument);
  Identifier argumentName(const model::RawFunctionDefinition &Function,
                          const model::Register::Values &Register) {
    return argumentName(Function, Function.Arguments().at(Register));
  }

  Identifier returnValueName(const model::RawFunctionDefinition &Function,
                             const model::NamedTypedRegister &ReturnValue);
  Identifier returnValueName(const model::RawFunctionDefinition &Function,
                             const model::Register::Values &Register) {
    return returnValueName(Function, Function.ReturnValues().at(Register));
  }

public:
  Identifier artificialReturnValueWrapperName(const RFT &Function) {
    // Not checking anything here since the relevant checks should have already
    // been done when constructing the helper.
    auto R = configuration().artificialReturnValuePrefix() + name(Function);
    return Identifier(R.str());
  }

private:
  RecursiveCoroutine<std::string>
  artificialArrayWrapperNameImpl(const model::Type &Type);

public:
  Identifier artificialArrayWrapperName(const model::ArrayType &Type) {
    // Not checking anything here since the relevant checks should have already
    // been done when constructing the helper.
    std::string_view Prefix = configuration().artificialArrayWrapperPrefix();
    return Identifier(std::string(Prefix)
                      + std::string(artificialArrayWrapperNameImpl(Type)));
  }

  Identifier artificialArrayWrapperFieldName();
};

} // namespace model
