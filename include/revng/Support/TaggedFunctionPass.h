#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/SmallVector.h"
#include "llvm/Pass.h"

#include "revng/Support/Assert.h"

struct TaggedFunctionPass : public llvm::ModulePass {
private:
  FunctionTags::Tag *InputTag;
  FunctionTags::Tag *OutputTag;

public:
  TaggedFunctionPass(char &ID,
                     FunctionTags::Tag *InputTag,
                     FunctionTags::Tag *OutputTag) :
    llvm::ModulePass(ID), InputTag(InputTag), OutputTag(OutputTag) {}

  bool runOnModule(llvm::Module &M) override {
    using namespace llvm;

    bool Changed = false;

    llvm::SmallVector<Function *> Functions;
    for (Function &F : InputTag->functions(&M))
      Functions.push_back(&F);

    for (Function *F : Functions) {
      if (not F->isDeclaration())
        Changed = runOnFunction(*F) or Changed;

      if (OutputTag != nullptr)
        OutputTag->addTo(F);

      Changed = true;
    }

    return Changed;
  }

public:
  virtual bool runOnFunction(llvm::Function &F) = 0;
};
