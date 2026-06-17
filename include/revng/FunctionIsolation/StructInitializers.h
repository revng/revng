#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Support/IRBuilder.h"
#include "revng/Support/OpaqueFunctionsPool.h"

namespace revng {
class IRBuilder;
} // namespace revng

class StructInitializers {
private:
  OpaqueFunctionsPool<llvm::StructType *> Pool;
  llvm::LLVMContext &Context;
  bool EmitBody = true;

public:
  StructInitializers(llvm::Module *M, bool EmitBody = true);

public:
  llvm::CallInst *createCall(revng::IRBuilder &Builder,
                             llvm::StructType *ReturnType,
                             llvm::ArrayRef<llvm::Value *> Values);

  llvm::Instruction *createReturn(revng::IRBuilder &Builder,
                                  llvm::ArrayRef<llvm::Value *> Values);
};
