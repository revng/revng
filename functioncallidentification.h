#ifndef _FUNCTIONCALLIDENTIFICATION_H
#define _FUNCTIONCALLIDENTIFICATION_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// LLVM includes
#include "llvm/Pass.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Casting.h"

// Local includes
#include "generatedcodebasicinfo.h"
#include "ir-helpers.h"

/// \brief Identify function call instructions
///
/// This pass parses the generated IR looking for terminator instructions which
/// look like function calls, i.e., they store the next program counter (the
/// return address) to a CSV (the link register) or in memory (the stack).
///
/// The pass also injects a call to the "funcion_call" function before
/// terminator instructions identified. The first argument represents the callee
/// basic block, the second the return basic block and the third the return
/// address.
class FunctionCallIdentification : public llvm::FunctionPass {
public:
  static char ID;

public:
  FunctionCallIdentification() : llvm::FunctionPass(ID) { }

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    AU.addRequired<GeneratedCodeBasicInfo>();
  }

  bool runOnFunction(llvm::Function &F) override;

  /// \brief Return true if \p T is a function call in the input assembly
  bool isCall(llvm::TerminatorInst *T) const {
    assert(T != nullptr);
    llvm::Instruction *Previous = getPrevious(T);
    if (Previous == nullptr)
      return false;

    if (auto *Call = llvm::dyn_cast<llvm::CallInst>(Previous))
      return Call->getCalledFunction() == FunctionCall;

    return false;
  }

  bool isCall(llvm::BasicBlock *BB) const {
    return isCall(BB->getTerminator());
  }

private:
  llvm::Function *FunctionCall;
};

#endif // _FUNCTIONCALLIDENTIFICATION_H
