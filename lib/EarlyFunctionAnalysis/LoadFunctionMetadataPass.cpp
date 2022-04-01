/// \file LoadFunctionMetadataPass.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/EarlyFunctionAnalysis/IRHelpers.h"
#include "revng/EarlyFunctionAnalysis/LoadFunctionMetadataPass.h"
#include "revng/Model/IRHelpers.h"

const efa::FunctionMetadata &LoadFunctionMetadataPass::get(llvm::Function *F) {
  auto Address = getMetaAddressOfIsolatedFunction(*F);
  auto It = EntryToMetadata.find(Address);
  if (It == EntryToMetadata.end()) {
    efa::FunctionMetadata FM = *extractFunctionMetadata(F).get();
    It = EntryToMetadata.insert(It, { Address, std::move(FM) });
  }

  return It->second;
}

bool LoadFunctionMetadataPass::runOnModule(llvm::Module &M) {
  // Do nothing
  return false;
}
