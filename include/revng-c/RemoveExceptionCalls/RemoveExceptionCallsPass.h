#ifndef REVNGC_REMOVE_EXCEPTION_CALLS_H
#define REVNGC_REMOVE_EXCEPTION_CALLS_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
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
