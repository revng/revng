/// \file SerializeModelPass.cpp
/// \brief Implementation of the pass taking care of serializing the
///        model in the Module as Metadata.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// LLVM includes
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"

// Local libraries includes
#include "revng/Model/SerializeModelPass.h"

using namespace llvm;

char SerializeModelWrapperPass::ID;

template<typename T>
using RP = RegisterPass<T>;

static RP<SerializeModelWrapperPass>
  X("serialize-model", "Serialize the model", true, true);

void writeModel(const model::Binary &Model, llvm::Module &M) {
  Model.verify(true);

  NamedMDNode *NamedMD = M.getNamedMetadata(ModelMetadataName);
  revng_check(not NamedMD, "The model has alread been serialized");

  std::string Buffer;
  {
    llvm::raw_string_ostream Stream(Buffer);
    serialize(Stream, Model);
  }

  LLVMContext &Context = M.getContext();
  auto Tuple = MDTuple::get(Context, { MDString::get(Context, Buffer) });

  NamedMD = M.getOrInsertNamedMetadata(ModelMetadataName);
  NamedMD->addOperand(Tuple);
}

bool SerializeModelWrapperPass::runOnModule(Module &M) {
  auto LoadPass = getAnalysisIfAvailable<LoadModelWrapperPass>();
  if (not LoadPass)
    return false;

  if (not ModelPath.empty()) {
    llvm::cantFail(LoadPass->get().getWriteableModel().toFile(ModelPath));
    return false;
  }

  writeModel(*LoadPass->get().getWriteableModel(), M);
  return false;
}

llvm::PreservedAnalyses
SerializeModelPass::run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM) {
  auto *ModelWrapper = MAM.getCachedResult<LoadModelAnalysis>(M);
  if (not ModelWrapper)
    return PreservedAnalyses::all();

  writeModel(*ModelWrapper->getWriteableModel(), M);
  return PreservedAnalyses::all();
}
