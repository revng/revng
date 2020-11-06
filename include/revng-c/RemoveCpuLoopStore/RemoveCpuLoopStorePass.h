#ifndef REVNGC_REMOVE_CPU_LOOP_STORE_H
#define REVNGC_REMOVE_CPU_LOOP_STORE_H

//
// Copyright (c) rev.ng Srls 2017-2020.
//

// LLVM includes
#include <llvm/Pass.h>

class RemoveCpuLoopStorePass : public llvm::FunctionPass {
public:
  static char ID;

public:
  RemoveCpuLoopStorePass() : llvm::FunctionPass(ID) {}

  bool runOnFunction(llvm::Function &F) override;
};

#endif // REVNGC_REMOVE_CPU_LOOP_STORE_H
