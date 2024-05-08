#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include "llvm/IR/IRBuilder.h"

#include "revng/Support/OpaqueFunctionsPool.h"

class StructInitializers {
private:
  OpaqueFunctionsPool<llvm::StructType *> Pool;
  llvm::LLVMContext &Context;

public:
  StructInitializers(llvm::Module *M);

public:
  llvm::Instruction *createReturn(llvm::IRBuilder<> &Builder,
                                  llvm::ArrayRef<llvm::Value *> Values);
};
