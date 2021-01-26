/// \file SerializeModelPass.cpp
/// \brief Implementation of the pass taking care of serializing the
///        model in the Module as Metadata.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// LLVM includes
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/PassSupport.h"

// Local libraries includes
#include "revng/Model/SerializeModelPass.h"

using namespace llvm;

char SerializeModelPass::ID;

template<typename T>
using RP = RegisterPass<T>;

static RP<SerializeModelPass>
  X("serialize-model", "Serialize the model", true, true);

bool SerializeModelPass::runOnModule(Module &M) {
  auto &LMP = getAnalysis<LoadModelPass>();

  LLVMContext &Context = M.getContext();
  std::string Buffer;
  {
    llvm::raw_string_ostream Stream(Buffer);
    yaml::Output YAMLOutput(Stream);
    YAMLOutput << LMP.getWriteableModel();
  }

  NamedMDNode *NamedMD = M.getNamedMetadata(ModelMetadataName);
  revng_assert(NamedMD == nullptr, "The model has alread been serialized");

  NamedMD = M.getOrInsertNamedMetadata(ModelMetadataName);
  auto Tuple = MDTuple::get(Context, { MDString::get(Context, Buffer) });
  NamedMD->addOperand(Tuple);

  return false;
}
