#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <map>
#include <tuple>

#include "llvm/IR/ModuleSlotTracker.h"

#include "revng/ADT/ConstantRangeSet.h"

namespace llvm {
class Value;
class Module;
} // namespace llvm

class DataFlowRangeAnalysis {
public:
  using CacheEntry = std::pair<llvm::Value *, llvm::Value *>;

private:
  std::map<CacheEntry, ConstantRangeSet> Cache;
  llvm::ModuleSlotTracker MST;

public:
  DataFlowRangeAnalysis(llvm::Module &M) : MST(&M, false) {}

public:
  std::optional<ConstantRangeSet> visit(llvm::Value &I, llvm::Value &Variable);
};
