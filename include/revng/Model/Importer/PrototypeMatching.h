#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Binary.h"

using ModelMap = std::map<std::string, TupleTree<model::Binary>>;

namespace detail {

template<typename T>
using TMS = TrackingMutableSet<T>;

template<typename T>
using TSV = TrackingSortedVector<T>;

} // namespace detail

struct FunctionInfo {
  const model::TypeDefinition &Prototype;
  const detail::TMS<model::FunctionAttribute::Values> &Attributes;
  llvm::StringRef Comment = {};
  llvm::StringRef ModuleName = {};
};

std::optional<FunctionInfo>
findPrototypeInLocalFunctions(const detail::TSV<model::Function> &Functions,
                              llvm::StringRef FunctionName,
                              llvm::StringRef ModuleName);

std::optional<FunctionInfo>
findPrototypeInDynamicFunctions(const detail::TSV<model::DynamicFunction>
                                  &Functions,
                                llvm::StringRef FunctionName,
                                llvm::StringRef ModuleName);

std::optional<FunctionInfo> findPrototype(llvm::StringRef Function,
                                          ModelMap &ModelsOfDynamicLibraries);
