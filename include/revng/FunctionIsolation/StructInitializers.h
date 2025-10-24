#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Support/OpaqueFunctionsPool.h"

namespace revng {
class IRBuilder;
} // namespace revng

class StructInitializers {
private:
  OpaqueFunctionsPool<llvm::StructType *> Pool;
  llvm::LLVMContext &Context;

public:
  StructInitializers(llvm::Module *M);

public:
  llvm::Instruction *createReturn(revng::IRBuilder &Builder,
                                  llvm::ArrayRef<llvm::Value *> Values);
};
