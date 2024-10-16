#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/Model/ArrayType.h"
#include "revng/Model/CABIFunctionDefinition.h"
#include "revng/Model/EnumDefinition.h"
#include "revng/Model/Identifier.h"
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

class VerifyHelper;

class NamingHelper {
private:
  using GlobalNameCacheType = std::unordered_map<std::string,
                                                 std::string,
                                                 detail::TransparentStringHash,
                                                 std::equal_to<>>;

  GlobalNameCacheType GlobalNameCache = {};

public:
  const NamingConfiguration &Configuration;

private:
  NamingHelper(const NamingConfiguration &Configuration) :
    Configuration(Configuration) {}

  std::optional<std::string_view> isNameForbidden(std::string_view Name);
  bool populateGlobalNameCache(VerifyHelper &VH, const Binary &Binary);

public:
  static std::optional<NamingHelper> tryMake(VerifyHelper &VH,
                                             const Binary &Binary);
  NamingHelper(const Binary &Binary);

public:
  [[nodiscard]] bool isGlobalSymbol(std::string_view Name) const {
    return GlobalNameCache.contains(Name);
  }

public:
  Identifier segment(const model::Segment &Segment) const;
  Identifier function(const model::Function &Function) const;
  Identifier dynamicFunction(const model::DynamicFunction &Function) const;

  std::string LLVMFunction(const model::Function &Function) const {
    std::string Result = "local_";
    if (DebugNames)
      Result += function(Function).str().str();
    else
      Result += Function.Entry().toString();

    return Result;
  }

public:
  Identifier type(const model::TypeDefinition &Definition) const;

  Identifier enumEntry(const model::EnumEntry &Entry,
                       const model::EnumDefinition &Definition) const;
  Identifier enumEntry(const uint64_t &Value,
                       const model::EnumDefinition &Definition) const {
    return enumEntry(Definition.Entries().at(Value), Definition);
  }

  Identifier field(const model::StructField &Field,
                   const model::StructDefinition &Definition) const;
  Identifier field(const uint64_t &Offset,
                   const model::StructDefinition &Definition) const {
    revng_assert(Offset <= Definition.Size());
    return field(Definition.Fields().at(Offset), Definition);
  }

  Identifier field(const model::UnionField &Field,
                   const model::UnionDefinition &Definition) const;
  Identifier field(const uint64_t &Index,
                   const model::UnionDefinition &Definition) const {
    revng_assert(Index <= Definition.Fields().size());
    return field(Definition.Fields().at(Index), Definition);
  }

  Identifier padding(const uint64_t &Offset) const {
    // Not checking anything here since the relevant checks should have already
    // been done when constructing the helper.
    return Identifier(std::string(Configuration.structPaddingPrefix())
                      + std::to_string(Offset));
  }

public:
  Identifier argument(const model::Argument &Argument,
                      const model::CABIFunctionDefinition &Function) const;
  Identifier argument(const uint64_t &ArgumentIndex,
                      const model::CABIFunctionDefinition &Function) const {
    revng_assert(ArgumentIndex <= Function.Arguments().size());
    return argument(Function.Arguments().at(ArgumentIndex), Function);
  }

  Identifier argument(const model::NamedTypedRegister &Argument,
                      const model::RawFunctionDefinition &Function) const;
  Identifier argument(const model::Register::Values &Register,
                      const model::RawFunctionDefinition &Function) const {
    return argument(Function.Arguments().at(Register), Function);
  }

  Identifier returnValue(const model::NamedTypedRegister &ReturnValue,
                         const model::RawFunctionDefinition &Function) const;
  Identifier returnValue(const model::Register::Values &Register,
                         const model::RawFunctionDefinition &Function) const {
    return argument(Function.ReturnValues().at(Register), Function);
  }

private:
  using RFT = model::RawFunctionDefinition;

public:
  Identifier artificialReturnValueWrapper(const RFT &Function) const {
    // Not checking anything here since the relevant checks should have already
    // been done when constructing the helper.
    auto Result = Configuration.artificialReturnValuePrefix() + type(Function);
    return Identifier(Result.str());
  }

private:
  RecursiveCoroutine<std::string>
  artificialArrayWrapperImpl(const model::Type &Type) const;

public:
  Identifier artificialArrayWrapper(const model::ArrayType &Type) const {
    // Not checking anything here since the relevant checks should have already
    // been done when constructing the helper.
    return Identifier(std::string(Configuration.artificialArrayWrapperPrefix())
                      + std::string(artificialArrayWrapperImpl(Type)));
  }

  Identifier artificialArrayWrapperFieldName() const;
};

} // namespace model
