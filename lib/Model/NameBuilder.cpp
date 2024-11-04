/// \file NameBuilder.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Binary.h"

const model::NamingConfiguration &model::NameBuilder::configuration() const {
  return Binary.Configuration().Naming();
}

static llvm::Error makeDuplicateSymbolError(const std::string &Name,
                                            const std::string &FirstPath,
                                            const std::string &SecondPath) {
  std::string Result;
  Result += "Duplicate global symbol \"";
  Result += Name;
  Result += "\":\n\n";

  Result += "  " + FirstPath + "\n";
  Result += "  " + SecondPath + "\n";
  return revng::createError(std::move(Result));
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

llvm::Error model::NameBuilder::isNameForbidden(std::string_view Name) {
  // Do these strictly for now, then soften them if needed.

  if (Name.starts_with(configuration().structPaddingPrefix()))
    return revng::createError("it is reserved for struct padding");

  if (Name.starts_with(configuration().artificialReturnValuePrefix()))
    return revng::createError("it is reserved for artificial return value "
                              "structs");

  if (Name.starts_with(configuration().artificialArrayWrapperPrefix()))
    return revng::createError("it is reserved for artificial array wrappers");

  // TODO: more of these?

  return llvm::Error::success();
}

llvm::Error model::NameBuilder::populateGlobalNamespace() {
  revng_assert(not GlobalNamespace.has_value());
  GlobalNamespace = GlobalNamespaceMap();

  auto RegisterGlobalSymbol = [this](std::string Name,
                                     std::string Path) -> llvm::Error {
    if (not Name.empty()) {
      if (llvm::Error Error = isNameForbidden(Name)) {
        struct ExtractMessage {
          std::string &Out;

          llvm::Error operator()(const llvm::StringError &Error) {
            Out = Error.getMessage();
            return llvm::Error::success();
          }
        };
        auto CatchAll = [](const llvm::ErrorInfoBase &) -> llvm::Error {
          revng_abort("Unsupported error type.");
        };

        std::string Out;
        llvm::handleAllErrors(std::move(Error),
                              ExtractMessage{ Out },
                              CatchAll);

        revng_assert(not Out.empty());
        return revng::createError("The name \"" + Name + "\" of \"" + Path
                                  + "\" is not allowed: " + std::move(Out)
                                  + ".\n");
      }

      auto [Iterator, Success] = GlobalNamespace->try_emplace(Name, Path);
      if (not Success)
        return makeDuplicateSymbolError(Name, Iterator->second, Path);
    }

    return llvm::Error::success();
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
    if (auto Error = RegisterGlobalSymbol(F.CustomName().str().str(), path(F)))
      return Error;

  for (const DynamicFunction &DF : Binary.ImportedDynamicFunctions())
    if (auto Error = RegisterGlobalSymbol(DF.CustomName().str().str(),
                                          path(DF)))
      return Error;

  for (const model::UpcastableTypeDefinition &D : Binary.TypeDefinitions()) {
    if (auto Error = RegisterGlobalSymbol(D->CustomName().str().str(),
                                          path(*D)))
      return Error;

    if (auto *Enum = llvm::dyn_cast<model::EnumDefinition>(D.get()))
      for (auto &Entry : Enum->Entries())
        if (auto Error = RegisterGlobalSymbol(Entry.CustomName().str().str(),
                                              path(*Enum, Entry)))
          return Error;
  }

  for (const Segment &S : Binary.Segments())
    if (auto Error = RegisterGlobalSymbol(S.CustomName().str().str(), path(S)))
      return Error;

  return llvm::Error::success();
}

static std::string toIdentifier(const MetaAddress &Address) {
  return model::Identifier::sanitize(Address.toString()).str().str();
}

model::Identifier model::NameBuilder::name(const model::Segment &Segment) {
  if (not Segment.CustomName().empty()) {
    revng_assert(globalNamespace().contains(Segment.CustomName()),
                 "Cache is outdated.");
    return Segment.CustomName();

  } else {
    auto Result = std::string(configuration().unnamedSegmentPrefix())
                  + toIdentifier(Segment.StartAddress()) + "_"
                  + std::to_string(Segment.VirtualSize());
    while (globalNamespace().contains(Result))
      Result += configuration().collisionResolutionSuffix();

    return Identifier(Result);
  }
}

model::Identifier model::NameBuilder::name(const model::Function &Function) {
  if (not Function.CustomName().empty()) {
    revng_assert(globalNamespace().contains(Function.CustomName()),
                 "Cache is outdated.");
    return Function.CustomName();

  } else {
    auto Result = std::string(configuration().unnamedFunctionPrefix())
                  + toIdentifier(Function.Entry());
    while (globalNamespace().contains(Result))
      Result += configuration().collisionResolutionSuffix();

    return Identifier(Result);
  }
}

model::Identifier
model::NameBuilder::name(const model::DynamicFunction &Function) {
  if (not Function.CustomName().empty()) {
    revng_assert(globalNamespace().contains(Function.CustomName()),
                 "Cache is outdated.");
    return Function.CustomName();

  } else {
    auto Result = std::string(configuration().unnamedDynamicFunctionPrefix())
                  + Function.OriginalName();
    while (globalNamespace().contains(Result))
      Result += configuration().collisionResolutionSuffix();

    return Identifier(Result);
  }
}

model::Identifier
model::NameBuilder::name(const model::TypeDefinition &Definition) {
  if (not Definition.CustomName().empty()) {
    revng_assert(globalNamespace().contains(Definition.CustomName()),
                 "Cache is outdated.");
    return Definition.CustomName();

  } else {
    auto K = model::TypeDefinitionKind::automaticNamePrefix(Definition.Kind());
    auto Result = std::string(configuration().unnamedTypeDefinitionPrefix())
                  + K.str() + std::to_string(Definition.ID());
    while (globalNamespace().contains(Result))
      Result += configuration().collisionResolutionSuffix();

    return Identifier(Result);
  }
}

model::Identifier
model::NameBuilder::name(const model::EnumDefinition &Definition,
                         const model::EnumEntry &Entry) {
  revng_assert(Definition.Entries().count(Entry.Value()) != 0);

  model::Identifier Result;

  // Decide on a name
  if (Entry.CustomName().empty()) {
    llvm::StringRef Prefix{ configuration().unnamedEnumEntryPrefix() };
    (Prefix + name(Definition) + llvm::Twine(Entry.Value())).toVector(Result);

  } else {
    Result = Entry.CustomName();
    revng_assert(globalNamespace().contains(Result), "Cache is outdated.");
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
    return globalNamespace().contains(Name);
  };
  while (NameCollides(Result))
    Result += configuration().collisionResolutionSuffix();

  return Result;
}

model::Identifier
model::NameBuilder::name(const model::StructDefinition &Definition,
                         const model::StructField &Field) {
  model::Identifier Result;

  // Decide on a name
  if (Field.CustomName().empty()) {
    llvm::StringRef Prefix{ configuration().unnamedStructFieldPrefix() };
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
    return globalNamespace().contains(Name);
  };
  while (NameCollides(Result))
    Result += configuration().collisionResolutionSuffix();

  return Result;
}

model::Identifier
model::NameBuilder::name(const model::UnionDefinition &Definition,
                         const model::UnionField &Field) {
  model::Identifier Result;

  // Decide on a name
  if (Field.CustomName().empty()) {
    llvm::StringRef Prefix{ configuration().unnamedUnionFieldPrefix() };
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
    return globalNamespace().contains(Name);
  };
  while (NameCollides(Result))
    Result += configuration().collisionResolutionSuffix();

  return Result;
}

using CFT = model::CABIFunctionDefinition;
model::Identifier
model::NameBuilder::argumentName(const CFT &Function,
                                 const model::Argument &Argument) {
  model::Identifier Result;

  // Decide on a name
  if (Argument.CustomName().empty()) {
    llvm::StringRef Prefix{ configuration().unnamedFunctionArgumentPrefix() };
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
    return globalNamespace().contains(Name);
  };
  while (NameCollides(Result))
    Result += configuration().collisionResolutionSuffix();

  return Result;
}

using NTR = model::NamedTypedRegister;
using RFT = model::RawFunctionDefinition;
model::Identifier model::NameBuilder::argumentName(const RFT &Function,
                                                   const NTR &Argument) {
  model::Identifier Result;

  // Decide on a name
  if (Argument.CustomName().empty()) {
    llvm::StringRef Prefix = configuration().unnamedFunctionRegisterPrefix();
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
    return globalNamespace().contains(Name);
  };
  while (NameCollides(Result))
    Result += configuration().collisionResolutionSuffix();

  return Result;
}

using NTR = model::NamedTypedRegister;
using RFT = model::RawFunctionDefinition;
model::Identifier model::NameBuilder::returnValueName(const RFT &Function,
                                                      const NTR &ReturnValue) {
  model::Identifier Result;

  // Decide on a name
  if (ReturnValue.CustomName().empty()) {
    llvm::StringRef Prefix = configuration().unnamedFunctionRegisterPrefix();
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
    return globalNamespace().contains(Name);
  };
  while (NameCollides(Result))
    Result += configuration().collisionResolutionSuffix();

  return Result;
}

using T = model::Type;
RecursiveCoroutine<std::string>
model::NameBuilder::artificialArrayWrapperNameImpl(const T &Type) {
  if (auto *Array = llvm::dyn_cast<model::ArrayType>(&Type)) {
    std::string Result = "array_" + std::to_string(Array->ElementCount())
                         + "_of_";
    Result += rc_recur artificialArrayWrapperNameImpl(*Array->ElementType());
    rc_return Result;

  } else if (auto *D = llvm::dyn_cast<model::DefinedType>(&Type)) {
    std::string Result = (D->IsConst() ? "const_" : "");
    rc_return std::move(Result += this->name(D->unwrap()).str().str());

  } else if (auto *Pointer = llvm::dyn_cast<model::PointerType>(&Type)) {
    std::string Result = (D->IsConst() ? "const_ptr_to_" : "ptr_to_");
    Result += rc_recur artificialArrayWrapperNameImpl(*Pointer->PointeeType());
    rc_return std::move(Result);

  } else if (auto *Primitive = llvm::dyn_cast<model::PrimitiveType>(&Type)) {
    std::string Result = (D->IsConst() ? "const_" : "");
    rc_return std::move(Result += Primitive->getCName());

  } else {
    revng_abort("Unsupported model::Type.");
  }
}

model::Identifier model::NameBuilder::artificialArrayWrapperFieldName() {
  std::string Result{ configuration().artificialArrayWrapperFieldName() };
  while (globalNamespace().contains(Result))
    Result += configuration().collisionResolutionSuffix();

  return Identifier(Result);
}
