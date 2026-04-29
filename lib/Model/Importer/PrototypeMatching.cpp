//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Importer/PrototypeMatching.h"

using FunctionVector = TrackingSortedVector<model::Function>;
using DynamicFunctionVector = TrackingSortedVector<model::DynamicFunction>;

std::optional<FunctionInfo>
findPrototypeInLocalFunctions(const FunctionVector &Functions,
                              llvm::StringRef FunctionName,
                              llvm::StringRef ModuleName) {
  for (auto &Function : Functions) {
    if (not llvm::is_contained(Function.ExportedNames(), FunctionName))
      continue;

    if (const model::TypeDefinition *Prototype = Function.prototype())
      return FunctionInfo{ .Prototype = *Prototype,
                           .Attributes = Function.Attributes(),
                           .ModuleName = ModuleName };
  }

  return std::nullopt;
}

std::optional<FunctionInfo>
findPrototypeInDynamicFunctions(const DynamicFunctionVector &Functions,
                                llvm::StringRef FunctionName,
                                llvm::StringRef ModuleName) {
  auto It = Functions.find(FunctionName.str());
  if (It == Functions.end())
    return std::nullopt;

  if (const model::TypeDefinition *Prototype = It->prototype())
    return FunctionInfo{ .Prototype = *Prototype,
                         .Attributes = It->Attributes(),
                         .ModuleName = ModuleName };

  return std::nullopt;
}

std::optional<FunctionInfo> findPrototype(llvm::StringRef Function,
                                          ModelMap &ModelsOfDynamicLibraries) {
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
