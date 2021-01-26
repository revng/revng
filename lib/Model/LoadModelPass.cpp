/// \file LoadModelPass.cpp
/// \brief Implementation of the immutable pass providing access to
///        the model and taking care of its deserialization on the IR.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// LLVM includes
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/PassSupport.h"

// Local libraries includes
#include "revng/Model/LoadModelPass.h"

using namespace llvm;

char LoadModelPass::ID;

template<typename T>
using RP = RegisterPass<T>;

static RP<LoadModelPass> X("load-model", "Deserialize the model", true, true);

bool LoadModelPass::doInitialization(Module &M) {
  NamedMDNode *NamedMD = M.getNamedMetadata(ModelMetadataName);
  if (NamedMD == nullptr or NamedMD->getNumOperands() < 1)
    return false;

  auto *Tuple = cast<MDTuple>(NamedMD->getOperand(0));
  if (Tuple->getNumOperands() < 1)
    return false;

  Metadata *MD = Tuple->getOperand(0).get();
  StringRef YAMLString = cast<MDString>(MD)->getString();

  yaml::Input YAMLInput(YAMLString);
  YAMLInput >> TheBinary;

  // Erase the named metadata in order to make sure no one is tempted to
  // deserialize it on its own
  NamedMD->eraseFromParent();

  return false;
}

bool LoadModelPass::doFinalization(Module &M) {
  if (not Modified)
    return false;

  // Check if the named metadata has reappeared. If not, the changes we made in
  // this pipeline would go lost
  NamedMDNode *NamedMD = M.getNamedMetadata(ModelMetadataName);
  revng_check(NamedMD != nullptr,
              "The model has changed, but -serialize-model has not been run");

  return false;
}
