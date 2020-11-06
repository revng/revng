#ifndef REVNGC_REMOVE_EXCEPTION_CALLS_H
#define REVNGC_REMOVE_EXCEPTION_CALLS_H

//
// Copyright (c) rev.ng Srls 2017-2020.
//

// LLVM includes
#include <llvm/Pass.h>

class RemoveExceptionCallsPass : public llvm::FunctionPass {
public:
  static char ID;

public:
  RemoveExceptionCallsPass() : llvm::FunctionPass(ID) {}

  bool runOnFunction(llvm::Function &F) override;
};

#endif // REVNGC_REMOVE_EXCEPTION_CALLS_H
