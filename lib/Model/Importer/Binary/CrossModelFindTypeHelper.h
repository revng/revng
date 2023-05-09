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
using ModelFunctionAttributes = TrackingMutableSet<
  model::FunctionAttribute::Values>;

template<typename T>
std::optional<std::pair<model::TypePath, ModelFunctionAttributes>>
findPrototypeInLocalFunctions(T &Functions, llvm::StringRef FunctionName) {
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

    if (!Function.Prototype().isValid())
      continue;

    return std::make_pair(Function.Prototype(), Function.Attributes());
  }

  return std::nullopt;
}

template<typename T>
std::optional<std::pair<model::TypePath, ModelFunctionAttributes>>
findPrototypeInDynamicFunctions(T &Functions, llvm::StringRef FunctionName) {
  for (auto &DynamicFunction : Functions) {
    // Rely on OriginalName only.
    if (DynamicFunction.OriginalName() != FunctionName)
      continue;

    if (!DynamicFunction.Prototype().isValid())
      continue;

    return std::make_pair(DynamicFunction.Prototype(),
                          DynamicFunction.Attributes());
  }

  return std::nullopt;
}

// This represents information about dynamic function we are about to copy into
// the Model.
struct FunctionInfo {
  model::TypePath Type;
  ModelFunctionAttributes Attributes;
  std::string ModuleName;
};

std::optional<FunctionInfo> findPrototype(llvm::StringRef FunctionName,
                                          ModelMap &ModelsOfDynamicLibraries) {
  for (auto &ModelOfDep : ModelsOfDynamicLibraries) {
    auto Prototype = findPrototypeInLocalFunctions(ModelOfDep.second
                                                     ->Functions(),
                                                   FunctionName);
    if (Prototype)
      return FunctionInfo{ (*Prototype).first,
                           (*Prototype).second,
                           ModelOfDep.first };

    Prototype = findPrototypeInDynamicFunctions(ModelOfDep.second
                                                  ->ImportedDynamicFunctions(),
                                                FunctionName);
    if (Prototype)
      return FunctionInfo{ (*Prototype).first,
                           (*Prototype).second,
                           ModelOfDep.first };
  }

  return std::nullopt;
}
} // namespace
