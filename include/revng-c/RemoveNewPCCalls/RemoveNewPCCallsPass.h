#ifndef REVNGC_REMOVE_NEW_PC_CALLS_H
#define REVNGC_REMOVE_NEW_PC_CALLS_H

//
// Copyright (c) rev.ng Srls 2017-2020.
//

// LLVM includes
#include <llvm/Pass.h>

class RemoveNewPCCallsPass : public llvm::FunctionPass {
public:
  static char ID;

public:
  RemoveNewPCCallsPass() : llvm::FunctionPass(ID) {}

  bool runOnFunction(llvm::Function &F) override;
};

#endif // REVNGC_REMOVE_NEW_PC_CALLS_H
