/// \file NamingHelper.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Binary.h"

std::optional<model::NamingHelper>
model::NamingHelper::tryMake(VerifyHelper &VH, const Binary &Binary) {
  NamingHelper Result = NamingHelper(Binary.Configuration().Naming());

  if (not Result.populateGlobalNameCache(VH, Binary))
    return std::nullopt;

  return Result;
}
model::NamingHelper::NamingHelper(const Binary &Binary) :
  NamingHelper(Binary.Configuration().Naming()) {

  VerifyHelper VH(true);
  populateGlobalNameCache(VH, Binary);
}

static std::string makeDuplicateSymbolError(const std::string &Name,
                                            const std::string &FirstPath,
                                            const std::string &SecondPath) {
  std::string Result;
  Result += "Duplicate global symbol \"";
  Result += Name;
  Result += "\":\n\n";

  Result += "  " + FirstPath + "\n";
  Result += "  " + SecondPath + "\n";
  return Result;
}

template<typename T>
static std::string key(const T &Object) {
  return getNameFromYAMLScalar(KeyedObjectTraits<T>::key(Object));
}

static std::string path(const model::Function &F) {
  return "/Functions/" + key(F);
}

static std::string path(const model::DynamicFunction &F) {
  return "/ImportedDynamicFunctions/" + key(F);
}

static std::string path(const model::TypeDefinition &T) {
  return "/TypeDefinitions/" + key(T);
}

static std::string path(const model::EnumDefinition &D,
                        const model::EnumEntry &Entry) {
  return path(static_cast<const model::TypeDefinition &>(D))
         + "/EnumDefinition/Entries/" + key(Entry);
}

static std::string path(const model::Segment &Segment) {
  return "/Segments/" + key(Segment);
}

std::optional<std::string_view>
model::NamingHelper::isNameForbidden(std::string_view Name) {
  // Do these strictly for now, then soften them if needed.

  if (Name.starts_with(Configuration.structPaddingPrefix()))
    return "it is reserved for struct padding";

  if (Name.starts_with(Configuration.artificialReturnValuePrefix()))
    return "it is reserved for artificial return value structs";

  if (Name.starts_with(Configuration.artificialArrayWrapperPrefix()))
    return "it is reserved for artificial array wrappers";

  // TODO: more of these?

  return std::nullopt;
}

bool model::NamingHelper::populateGlobalNameCache(VerifyHelper &VH,
                                                  const Binary &Binary) {
  auto RegisterGlobalSymbol = [this, &VH](std::string Name,
                                          std::string Path) -> bool {
    if (not Name.empty()) {
      if (auto Error = isNameForbidden(Name))
        return VH.fail("The name \"" + Name + "\" of \"" + Path
                       + "\" is not allowed: " + std::string(*Error) + ".\n");

      auto [Iterator, Success] = GlobalNameCache.try_emplace(Name, Path);
      if (not Success)
        return VH.fail(makeDuplicateSymbolError(Name, Iterator->second, Path));
    }

    return true;
  };

  // Namespacing rules:
  //
  // 1. each struct/union induces a namespace for its field names;
  // 2. each prototype induces a namespace for its arguments (and local
  //    variables, but those are not part of the model yet);
  // 3. the global namespace includes segment names, function names, dynamic
  //    function names, type names, entries of `enum`s and helper names;
  //
  // Verify needs to verify that each namespace has no internal clashes.
  // Also, the global namespace clashes with everything.
  //
  // TODO: find a way to add all the helper names to the Result.
  for (const Function &F : Binary.Functions())
    if (not RegisterGlobalSymbol(F.CustomName().str().str(), path(F)))
      return false;

  for (const DynamicFunction &DF : Binary.ImportedDynamicFunctions())
    if (not RegisterGlobalSymbol(DF.CustomName().str().str(), path(DF)))
      return false;

  for (const model::UpcastableTypeDefinition &D : Binary.TypeDefinitions()) {
    if (not RegisterGlobalSymbol(D->CustomName().str().str(), path(*D)))
      return false;

    if (auto *Enum = llvm::dyn_cast<model::EnumDefinition>(D.get()))
      for (auto &Entry : Enum->Entries())
        if (not RegisterGlobalSymbol(Entry.CustomName().str().str(),
                                     path(*Enum, Entry)))
          return false;
  }

  for (const Segment &S : Binary.Segments())
    if (not RegisterGlobalSymbol(S.CustomName().str().str(), path(S)))
      return false;

  return true;
}

static std::string toIdentifier(const MetaAddress &Address) {
  return model::Identifier::sanitize(Address.toString()).str().str();
}

model::Identifier
model::NamingHelper::segment(const model::Segment &Segment) const {
  if (not Segment.CustomName().empty()) {
    revng_assert(GlobalNameCache.contains(Segment.CustomName()),
                 "Cache is outdated.");
    return Segment.CustomName();

  } else {
    auto Result = std::string(Configuration.unnamedSegmentPrefix())
                  + toIdentifier(Segment.StartAddress()) + "_"
                  + std::to_string(Segment.VirtualSize());
    while (GlobalNameCache.contains(Result))
      Result += Configuration.conflictResolutionSuffix();

    return Identifier(Result);
  }
}

model::Identifier
model::NamingHelper::function(const model::Function &Function) const {
  if (not Function.CustomName().empty()) {
    revng_assert(GlobalNameCache.contains(Function.CustomName()),
                 "Cache is outdated.");
    return Function.CustomName();

  } else {
    auto Result = std::string(Configuration.unnamedFunctionPrefix())
                  + toIdentifier(Function.Entry());
    while (GlobalNameCache.contains(Result))
      Result += Configuration.conflictResolutionSuffix();

    return Identifier(Result);
  }
}

model::Identifier
model::NamingHelper::dynamicFunction(const model::DynamicFunction &Function)
  const {
  if (not Function.CustomName().empty()) {
    revng_assert(GlobalNameCache.contains(Function.CustomName()),
                 "Cache is outdated.");
    return Function.CustomName();

  } else {
    auto Result = std::string(Configuration.unnamedDynamicFunctionPrefix())
                  + Function.OriginalName();
    while (GlobalNameCache.contains(Result))
      Result += Configuration.conflictResolutionSuffix();

    return Identifier(Result);
  }
}

model::Identifier
model::NamingHelper::type(const model::TypeDefinition &Definition) const {
  if (not Definition.CustomName().empty()) {
    revng_assert(GlobalNameCache.contains(Definition.CustomName()),
                 "Cache is outdated.");
    return Definition.CustomName();

  } else {
    auto K = model::TypeDefinitionKind::automaticNamePrefix(Definition.Kind());
    auto Result = std::string(Configuration.unnamedTypeDefinitionPrefix())
                  + K.str() + std::to_string(Definition.ID());
    while (GlobalNameCache.contains(Result))
      Result += Configuration.conflictResolutionSuffix();

    return Identifier(Result);
  }
}

model::Identifier
model::NamingHelper::enumEntry(const model::EnumEntry &Entry,
                               const model::EnumDefinition &Definition) const {
  revng_assert(Definition.Entries().count(Entry.Value()) != 0);

  model::Identifier Result;

  // Decide on a name
  if (Entry.CustomName().empty()) {
    llvm::StringRef Prefix{ Configuration.unnamedEnumEntryPrefix() };
    (Prefix + type(Definition) + llvm::Twine(Entry.Value())).toVector(Result);

  } else {
    Result = Entry.CustomName();
    revng_assert(GlobalNameCache.contains(Result), "Cache is outdated.");
    return Result;
  }

  // Ensure it doesn't collide with anything
  // TODO: factor these out?
  auto NameCollides = [this, &Entry, &Definition](const auto &Name) {
    for (const auto &AnotherEntry : Definition.Entries())
      if (Entry.key() != AnotherEntry.key())
        if (not AnotherEntry.CustomName().empty())
          if (AnotherEntry.CustomName() == Name)
            return true;
    return GlobalNameCache.contains(Name);
  };
  while (NameCollides(Result))
    Result += Configuration.conflictResolutionSuffix();

  // TODO: Are deeper checks necessary here?
  //       What if this knew value collides with something somewhere?

  return Result;
}

model::Identifier
model::NamingHelper::field(const model::StructField &Field,
                           const model::StructDefinition &Definition) const {
  model::Identifier Result;

  // Decide on a name
  if (Field.CustomName().empty()) {
    llvm::StringRef Prefix{ Configuration.unnamedStructFieldPrefix() };
    (Prefix + llvm::Twine(Field.Offset())).toVector(Result);

  } else {
    Result = Field.CustomName();
  }

  // Ensure it doesn't collide with anything
  // TODO: factor these out?
  auto NameCollides = [this, &Field, &Definition](const auto &Name) {
    for (const auto &AnotherField : Definition.Fields())
      if (Field.key() != AnotherField.key())
        if (not AnotherField.CustomName().empty())
          if (AnotherField.CustomName() == Name)
            return true;
    return GlobalNameCache.contains(Name);
  };
  while (NameCollides(Result))
    Result += Configuration.conflictResolutionSuffix();

  return Result;
}

model::Identifier
model::NamingHelper::field(const model::UnionField &Field,
                           const model::UnionDefinition &Definition) const {
  model::Identifier Result;

  // Decide on a name
  if (Field.CustomName().empty()) {
    llvm::StringRef Prefix{ Configuration.unnamedUnionFieldPrefix() };
    (Prefix + llvm::Twine(Field.Index())).toVector(Result);

  } else {
    Result = Field.CustomName();
  }

  // Ensure it doesn't collide with anything
  // TODO: factor these out?
  auto NameCollides = [this, &Field, &Definition](const auto &Name) {
    for (const auto &AnotherField : Definition.Fields())
      if (Field.key() != AnotherField.key())
        if (not AnotherField.CustomName().empty())
          if (AnotherField.CustomName() == Name)
            return true;
    return GlobalNameCache.contains(Name);
  };
  while (NameCollides(Result))
    Result += Configuration.conflictResolutionSuffix();

  return Result;
}

using CFT = model::CABIFunctionDefinition;
model::Identifier model::NamingHelper::argument(const model::Argument &Argument,
                                                const CFT &Function) const {
  model::Identifier Result;

  // Decide on a name
  if (Argument.CustomName().empty()) {
    llvm::StringRef Prefix{ Configuration.unnamedFunctionArgumentPrefix() };
    (Prefix + llvm::Twine(Argument.Index())).toVector(Result);

  } else {
    Result = Argument.CustomName();
  }

  // Ensure it doesn't collide with anything
  // TODO: factor these out?
  auto NameCollides = [this, &Argument, &Function](const auto &Name) {
    for (const auto &AnotherArgument : Function.Arguments())
      if (Argument.key() != AnotherArgument.key())
        if (not AnotherArgument.CustomName().empty())
          if (AnotherArgument.CustomName() == Name)
            return true;
    return GlobalNameCache.contains(Name);
  };
  while (NameCollides(Result))
    Result += Configuration.conflictResolutionSuffix();

  return Result;
}

using NTR = model::NamedTypedRegister;
using RFT = model::RawFunctionDefinition;
model::Identifier model::NamingHelper::argument(const NTR &Argument,
                                                const RFT &Function) const {
  model::Identifier Result;

  // Decide on a name
  if (Argument.CustomName().empty()) {
    llvm::StringRef Prefix = Configuration.unnamedFunctionRegisterPrefix();
    (Prefix + getRegisterName(Argument.Location())).toVector(Result);

  } else {
    Result = Argument.CustomName();
  }

  // Ensure it doesn't collide with anything
  // TODO: factor these out?
  auto NameCollides = [this, &Argument, &Function](const auto &Name) {
    for (const auto &AnotherArgument : Function.Arguments())
      if (Argument.key() != AnotherArgument.key())
        if (not AnotherArgument.CustomName().empty())
          if (AnotherArgument.CustomName() == Name)
            return true;
    return GlobalNameCache.contains(Name);
  };
  while (NameCollides(Result))
    Result += Configuration.conflictResolutionSuffix();

  return Result;
}

using NTR = model::NamedTypedRegister;
using RFT = model::RawFunctionDefinition;
model::Identifier model::NamingHelper::returnValue(const NTR &ReturnValue,
                                                   const RFT &Function) const {
  model::Identifier Result;

  // Decide on a name
  if (ReturnValue.CustomName().empty()) {
    llvm::StringRef Prefix = Configuration.unnamedFunctionRegisterPrefix();
    (Prefix + getRegisterName(ReturnValue.Location())).toVector(Result);

  } else {
    Result = ReturnValue.CustomName();
  }

  // Ensure it doesn't collide with anything
  // TODO: factor these out?
  auto NameCollides = [this, &ReturnValue, &Function](const auto &Name) {
    for (const auto &AnotherReturnValue : Function.ReturnValues())
      if (ReturnValue.key() != AnotherReturnValue.key())
        if (not AnotherReturnValue.CustomName().empty())
          if (AnotherReturnValue.CustomName() == Name)
            return true;
    return GlobalNameCache.contains(Name);
  };
  while (NameCollides(Result))
    Result += Configuration.conflictResolutionSuffix();

  return Result;
}

RecursiveCoroutine<std::string>
model::NamingHelper::artificialArrayWrapperImpl(const model::Type &Type) const {
  if (auto *Array = llvm::dyn_cast<model::ArrayType>(&Type)) {
    std::string Result = "array_" + std::to_string(Array->ElementCount())
                         + "_of_";
    Result += rc_recur artificialArrayWrapperImpl(*Array->ElementType());
    rc_return Result;

  } else if (auto *D = llvm::dyn_cast<model::DefinedType>(&Type)) {
    std::string Result = (D->IsConst() ? "const_" : "");
    rc_return std::move(Result += this->type(D->unwrap()).str().str());

  } else if (auto *Pointer = llvm::dyn_cast<model::PointerType>(&Type)) {
    std::string Result = (D->IsConst() ? "const_ptr_to_" : "ptr_to_");
    rc_return std::move(Result += rc_recur
                          artificialArrayWrapperImpl(*Pointer->PointeeType()));

  } else if (auto *Primitive = llvm::dyn_cast<model::PrimitiveType>(&Type)) {
    std::string Result = (D->IsConst() ? "const_" : "");
    rc_return std::move(Result += Primitive->getCName());

  } else {
    revng_abort("Unsupported model::Type.");
  }
}

model::Identifier model::NamingHelper::artificialArrayWrapperFieldName() const {
  std::string Result{ Configuration.artificialArrayWrapperFieldName() };
  while (GlobalNameCache.contains(Result))
    Result += Configuration.conflictResolutionSuffix();

  return Identifier(Result);
}
