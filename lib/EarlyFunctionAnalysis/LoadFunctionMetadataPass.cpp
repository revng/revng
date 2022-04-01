/// \file LoadFunctionMetadataPass.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/EarlyFunctionAnalysis/IRHelpers.h"
#include "revng/EarlyFunctionAnalysis/LoadFunctionMetadataPass.h"
#include "revng/Model/IRHelpers.h"

const efa::FunctionMetadata &LoadFunctionMetadataPass::get(llvm::Function *F) {
  auto Address = getMetaAddressOfIsolatedFunction(*F);
  if (EntryToMetadata.find(Address) == EntryToMetadata.end())
    EntryToMetadata.insert({ Address, *extractFunctionMetadata(F).get() });

  return EntryToMetadata[Address];
}

bool LoadFunctionMetadataPass::runOnModule(llvm::Module &M) {
  // Do nothing
  return false;
}
