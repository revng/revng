//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/Instruction.h"

#include "revng/Pipes/IRHelpers.h"
#include "revng/Support/IRHelpers.h"

llvm::DenseMap<MetaAddress, const llvm::Function *>
getTargetToFunctionMapping(const llvm::Module &M) {
  llvm::DenseMap<MetaAddress, const llvm::Function *> Map;
  for (const llvm::Function &F : M.functions()) {
    auto MA = getMetaAddressMetadata(&F, FunctionEntryMDName);
    if (MA.isValid()) {
      auto [Iterator, Inserted] = Map.try_emplace(MA, &F);
      revng_assert(Inserted);
    }
  }
  return Map;
}
