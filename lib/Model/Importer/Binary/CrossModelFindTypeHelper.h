#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <map>
#include <memory>
#include <set>

#include "revng/Model/Importer/TypeCopier.h"
#include "revng/Model/Processing.h"

namespace {
using ModelMap = std::map<std::string, TupleTree<model::Binary>>;

using TypeCopierMap = std::map<std::string, std::unique_ptr<TypeCopier>>;

// This represents information about dynamic function we are about to copy into
// the Model.
struct FunctionInfo {
  const model::TypeDefinition &Prototype;
  const TrackingMutableSet<model::FunctionAttribute::Values> &Attributes;
  llvm::StringRef ModuleName = {};
};

template<typename T>
std::optional<FunctionInfo>
findPrototypeInLocalFunctions(T &Functions,
                              llvm::StringRef FunctionName,
                              llvm::StringRef ModuleName) {
  for (auto &Function : Functions) {
    if (Function.ExportedNames().size()) {
      bool FoundAlias = false;
      for (auto &Name : Function.ExportedNames()) {
        if (Name == FunctionName) {
          FoundAlias = true;
        }
        if (FoundAlias)
          break;
        continue;
      }
      if (!FoundAlias)
        continue;
    } else {
      // Rely on OriginalName only.
      if (Function.OriginalName() != FunctionName)
        continue;
    }

    if (const model::TypeDefinition *Prototype = Function.prototype())
      return FunctionInfo{ .Prototype = *Prototype,
                           .Attributes = Function.Attributes(),
                           .ModuleName = ModuleName };
  }

  return std::nullopt;
}

template<typename T>
std::optional<FunctionInfo>
findPrototypeInDynamicFunctions(T &Functions,
                                llvm::StringRef FunctionName,
                                llvm::StringRef ModuleName) {
  for (auto &DynamicFunction : Functions) {
    // Rely on OriginalName only.
    if (DynamicFunction.OriginalName() != FunctionName)
      continue;

    if (const model::TypeDefinition *Prototype = DynamicFunction.prototype())
      return FunctionInfo{ .Prototype = *Prototype,
                           .Attributes = DynamicFunction.Attributes(),
                           .ModuleName = ModuleName };
  }

  return std::nullopt;
}

inline std::optional<FunctionInfo>
findPrototype(llvm::StringRef Function, ModelMap &ModelsOfDynamicLibraries) {
  for (const auto &[Module, Model] : ModelsOfDynamicLibraries) {
    const auto &Ls = Model->Functions();
    if (std::optional R = findPrototypeInLocalFunctions(Ls, Function, Module))
      return R;

    const auto &Ds = Model->ImportedDynamicFunctions();
    if (std::optional R = findPrototypeInDynamicFunctions(Ds, Function, Module))
      return R;
  }

  return std::nullopt;
}
} // namespace
