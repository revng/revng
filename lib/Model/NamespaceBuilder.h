#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Note that namespace collisions can only occur on invalid models (there are
// exactly two use cases of it - verification and post-importer fixes), as such
// there's no need to expose this logic outside of `libModel` - as it's expected
// that all the dependencies work exclusively on valid models.
//
// If you find yourself with a model containing duplicated identifiers after
// an analysis, instead of exposing this, take a look at `fix-model` pass (or
// one of its components) instead.

#include <map>
#include <unordered_set>

#include "revng/Model/Binary.h"
#include "revng/Model/TypePathHelpers.h"

namespace model {

template<bool IsConst, typename T>
using ConstIf = std::conditional_t<IsConst, const T, T>;

template<bool IsBinaryConst>
struct NamespaceEntry {
  using StoredNameType = ConstIf<IsBinaryConst, model::Identifier>;

  StoredNameType *Name;

  /// This is only used for improving error messages. It should be made optional
  /// if it ever impacts performance.
  std::string ModelPath;

  NamespaceEntry(StoredNameType &Name, std::string ModelPath) :
    Name(&Name), ModelPath(ModelPath) {}
};

template<bool IsConst>
using Namespace = std::map<llvm::StringRef,
                           std::vector<NamespaceEntry<IsConst>>>;

template<bool IsBinaryConst>
struct Namespaces {
  Namespace<IsBinaryConst> Global;
  std::vector<Namespace<IsBinaryConst>> Local;
};

/// Note that a valid model is guaranteed to not have any namespace collisions.
/// As such this logic is only needed when verifying (or fixing) a model.
template<ConstOrNot<model::Binary> BinaryType>
llvm::Expected<Namespaces<std::is_const_v<BinaryType>>>
collectNamespaces(BinaryType &Binary) {
  Namespaces<std::is_const_v<BinaryType>> Result;

  for (ConstOrNot<DynamicFunction> auto &F : Binary.ImportedDynamicFunctions())
    if (not F.CustomName().empty())
      Result.Global[F.CustomName()].emplace_back(F.CustomName(),
                                                 detail::path(F));

  // Dynamic functions are a bit special in that we cannot afford to change
  // their names no matter what. As such, if we run into any dynamic function
  // related collisions, we abort right away.
  std::string ProblemNameList;
  for (auto &&[Name, List] : Result.Global) {
    if (List.size() > 1) {
      ProblemNameList += "`" + Name.str() + "` (";
      for (const auto &Collision : List)
        ProblemNameList += *Collision.Name;
      ProblemNameList.resize(ProblemNameList.size() - 2);
      ProblemNameList += "), ";
    }
  }

  if (ProblemNameList.size() > 0) {
    ProblemNameList.resize(ProblemNameList.size() - 2);
    return revng::createError("Dynamic function names must never collide ("
                              + std::move(ProblemNameList) + ").");
  }

  for (ConstOrNot<Function> auto &F : Binary.Functions())
    if (not F.CustomName().empty())
      Result.Global[F.CustomName()].emplace_back(F.CustomName(),
                                                 detail::path(F));

  for (ConstOrNot<Segment> auto &S : Binary.Segments())
    if (not S.CustomName().empty())
      Result.Global[S.CustomName()].emplace_back(S.CustomName(),
                                                 detail::path(S));

  for (auto &Def : Binary.TypeDefinitions()) {
    if (not Def->CustomName().empty())
      Result.Global[Def->CustomName()].emplace_back(Def->CustomName(),
                                                    detail::path(*Def));

    if (auto *Enum = llvm::dyn_cast<model::EnumDefinition>(Def.get()))
      for (auto &Entry : Enum->Entries())
        if (not Entry.CustomName().empty())
          Result.Global[Entry.CustomName()].emplace_back(Entry.CustomName(),
                                                         detail::path(*Enum,
                                                                      Entry));
  }

  for (auto &Definition : Binary.TypeDefinitions()) {
    ConstOrNot<TypeDefinition> auto *D = Definition.get();
    if (llvm::isa<model::EnumDefinition>(D)) {
      // Skip enums since all their entries are a part of the global namespace

    } else if (llvm::isa<model::TypedefDefinition>(D)) {
      // Skip typedefs since they don't spawn a new namespace

    } else if (auto *RFT = llvm::dyn_cast<model::RawFunctionDefinition>(D)) {
      auto &Arguments = Result.Local.emplace_back();
      for (auto &Argument : RFT->Arguments())
        if (not Argument.CustomName().empty())
          Arguments[Argument.CustomName()].emplace_back(Argument.CustomName(),
                                                        detail::path(*RFT,
                                                                     Argument));

      auto &RVs = Result.Local.emplace_back();
      for (auto &ReturnValue : RFT->ReturnValues())
        if (not ReturnValue.CustomName().empty())
          RVs[ReturnValue.CustomName()].emplace_back(ReturnValue.CustomName(),
                                                     detail::path(*RFT,
                                                                  ReturnValue));

    } else if (auto *CFT = llvm::dyn_cast<model::CABIFunctionDefinition>(D)) {
      auto &Arguments = Result.Local.emplace_back();
      for (auto &Argument : CFT->Arguments())
        if (not Argument.CustomName().empty())
          Arguments[Argument.CustomName()].emplace_back(Argument.CustomName(),
                                                        detail::path(*CFT,
                                                                     Argument));

      // TODO: don't forget about local variables once those are in the model.

    } else if (auto *S = llvm::dyn_cast<model::StructDefinition>(D)) {
      auto &Fields = Result.Local.emplace_back();
      for (auto &Field : S->Fields())
        if (not Field.CustomName().empty())
          Fields[Field.CustomName()].emplace_back(Field.CustomName(),
                                                  detail::path(*S, Field));

    } else if (auto *U = llvm::dyn_cast<model::UnionDefinition>(D)) {
      auto &Fields = Result.Local.emplace_back();
      for (auto &Field : U->Fields())
        if (not Field.CustomName().empty())
          Fields[Field.CustomName()].emplace_back(Field.CustomName(),
                                                  detail::path(*U, Field));

    } else {
      revng_abort("Unsupported type definition kind.");
    }
  }

  return Result;
}

} // namespace model
