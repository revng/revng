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

char LoadModelWrapperPass::ID;

AnalysisKey LoadModelAnalysis::Key;

template<typename T>
using RP = RegisterPass<T>;

static RP<LoadModelWrapperPass>
  X("load-model", "Deserialize the model", true, true);

model::Binary loadModel(const llvm::Module &M) {

  model::Binary Result;

  NamedMDNode *NamedMD = M.getNamedMetadata(ModelMetadataName);
  revng_check(NamedMD and NamedMD->getNumOperands());

  auto *Tuple = cast<MDTuple>(NamedMD->getOperand(0));
  revng_check(Tuple->getNumOperands());

  Metadata *MD = Tuple->getOperand(0).get();
  StringRef YAMLString = cast<MDString>(MD)->getString();

  yaml::Input YAMLInput(YAMLString);
  YAMLInput >> Result;

  return Result;
}

static model::Binary extractModel(Module &M) {
  model::Binary Result = loadModel(M);

  // Erase the named metadata in order to make sure no one is tempted to
  // deserialize it on its own
  NamedMDNode *NamedMD = M.getNamedMetadata(ModelMetadataName);
  NamedMD->eraseFromParent();

  return Result;
}

bool LoadModelWrapperPass::doInitialization(Module &M) {
  Wrapper = extractModel(M);
  return false;
}

bool LoadModelWrapperPass::doFinalization(Module &M) {
  if (not Wrapper->hasChanged())
    return false;

  // Check if the named metadata has reappeared. If not, the changes we made in
  // this pipeline would go lost
  NamedMDNode *NamedMD = M.getNamedMetadata(ModelMetadataName);
  revng_check(NamedMD != nullptr,
              "The model has changed, but -serialize-model has not been run");

  return false;
}

ModelWrapper LoadModelAnalysis::run(Module &M, ModuleAnalysisManager &) {
  return { loadModel(M) };
}

ModelWrapper LoadModelAnalysis::run(Function &F, FunctionAnalysisManager &) {
  return { loadModel(*F.getParent()) };
}
