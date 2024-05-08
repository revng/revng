/// \file RemoveDbgMetadata.cpp
/// A simple pass to remove debug metadata from a function.

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"

#include "revng/BasicAnalyses/RemoveDbgMetadata.h"
#include "revng/Support/FunctionTags.h"

using namespace llvm;

char RemoveDbgMetadata::ID = 0;

using Register = RegisterPass<RemoveDbgMetadata>;
static Register X("remove-dbg-metadata", "Removes dbg metadata from Functions");

bool RemoveDbgMetadata::runOnFunction(llvm::Function &F) {
  if (not FunctionTags::Isolated.isTagOf(&F))
    return false;

  F.setMetadata(LLVMContext::MD_dbg, nullptr);
  for (BasicBlock &BB : F)
    for (Instruction &I : BB)
      I.setMetadata(LLVMContext::MD_dbg, nullptr);

  return true;
}
