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
  using StoredNameType = ConstIf<IsBinaryConst, std::string>;

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
    if (not F.Name().empty())
      Result.Global[F.Name()].emplace_back(F.Name(), detail::path(F));

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
    if (not F.Name().empty())
      Result.Global[F.Name()].emplace_back(F.Name(), detail::path(F));

  for (ConstOrNot<Segment> auto &S : Binary.Segments())
    if (not S.Name().empty())
      Result.Global[S.Name()].emplace_back(S.Name(), detail::path(S));

  for (auto &Def : Binary.TypeDefinitions()) {
    if (not Def->Name().empty())
      Result.Global[Def->Name()].emplace_back(Def->Name(), detail::path(*Def));

    if (auto *Enum = llvm::dyn_cast<model::EnumDefinition>(Def.get()))
      for (auto &Entry : Enum->Entries())
        if (not Entry.Name().empty())
          Result.Global[Entry.Name()].emplace_back(Entry.Name(),
                                                   detail::path(*Enum, Entry));
  }

  std::unordered_set<const model::TypeDefinition *> ProcessedPrototypes;
  for (ConstOrNot<Function> auto &Function : Binary.Functions()) {
    auto &Variables = Result.Local.emplace_back();
    for (ConstOrNot<LocalIdentifier> auto &Var : Function.LocalVariables()) {
      revng_assert(not Var.Name().empty());
      Variables[Var.Name()].emplace_back(Var.Name(),
                                         detail::variablePath(Function, Var));
    }

    for (ConstOrNot<LocalIdentifier> auto &Label : Function.GotoLabels()) {
      revng_assert(not Label.Name().empty());
      Variables[Label.Name()].emplace_back(Label.Name(),
                                           detail::gotoLabelPath(Function,
                                                                 Label));
    }

    if (auto *RFT = Function.rawPrototype()) {
      for (auto &Arg : RFT->Arguments())
        if (not Arg.Name().empty())
          Variables[Arg.Name()].emplace_back(Arg.Name(),
                                             detail::argumentPath(*RFT, Arg));

      ProcessedPrototypes.emplace(RFT);

    } else if (auto *CFT = Function.cabiPrototype()) {
      for (auto &Arg : CFT->Arguments())
        if (not Arg.Name().empty())
          Variables[Arg.Name()].emplace_back(Arg.Name(),
                                             detail::argumentPath(*CFT, Arg));

      ProcessedPrototypes.emplace(CFT);

    } else {
      revng_assert(Function.Prototype().isEmpty());
    }
  }

  for (auto &Definition : Binary.TypeDefinitions()) {
    ConstOrNot<TypeDefinition> auto *D = Definition.get();
    if (llvm::isa<model::EnumDefinition>(D)) {
      // Skip enums since all their entries are a part of the global namespace

    } else if (llvm::isa<model::TypedefDefinition>(D)) {
      // Skip typedefs since they don't spawn a new namespace

    } else if (auto *RFT = llvm::dyn_cast<model::RawFunctionDefinition>(D)) {
      if (not ProcessedPrototypes.contains(RFT)) {
        auto &Arguments = Result.Local.emplace_back();
        for (auto &Arg : RFT->Arguments())
          if (not Arg.Name().empty())
            Arguments[Arg.Name()].emplace_back(Arg.Name(),
                                               detail::argumentPath(*RFT, Arg));
      }

      auto &RVs = Result.Local.emplace_back();
      for (auto &ReturnV : RFT->ReturnValues())
        if (not ReturnV.Name().empty())
          RVs[ReturnV.Name()].emplace_back(ReturnV.Name(),
                                           detail::returnValuePath(*RFT,
                                                                   ReturnV));

    } else if (auto *CFT = llvm::dyn_cast<model::CABIFunctionDefinition>(D)) {
      if (not ProcessedPrototypes.contains(CFT)) {
        auto &Arguments = Result.Local.emplace_back();
        for (auto &Arg : CFT->Arguments())
          if (not Arg.Name().empty())
            Arguments[Arg.Name()].emplace_back(Arg.Name(),
                                               detail::argumentPath(*CFT, Arg));
      }

    } else if (auto *S = llvm::dyn_cast<model::StructDefinition>(D)) {
      auto &Fields = Result.Local.emplace_back();
      for (auto &Field : S->Fields())
        if (not Field.Name().empty())
          Fields[Field.Name()].emplace_back(Field.Name(),
                                            detail::path(*S, Field));

    } else if (auto *U = llvm::dyn_cast<model::UnionDefinition>(D)) {
      auto &Fields = Result.Local.emplace_back();
      for (auto &Field : U->Fields())
        if (not Field.Name().empty())
          Fields[Field.Name()].emplace_back(Field.Name(),
                                            detail::path(*U, Field));

    } else {
      revng_abort("Unsupported type definition kind.");
    }
  }

  return Result;
}

} // namespace model
