#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"

#include "revng/BasicAnalyses/CustomCFG.h"
#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/Support/IRHelpers.h"

/// Identify function call instructions
///
/// This pass parses the generated IR looking for terminator instructions which
/// look like function calls, i.e., they store the next program counter (the
/// return address) to a CSV (the link register) or in memory (the stack).
///
/// The pass also injects a call to the "funcion_call" function before
/// terminator instructions identified. The first argument represents the callee
/// basic block, the second the return basic block and the third the return
/// address.
class FunctionCallIdentification : public llvm::ModulePass {
public:
  static char ID;

public:
  FunctionCallIdentification() : llvm::ModulePass(ID) {}

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    AU.addRequired<GeneratedCodeBasicInfoWrapperPass>();
  }

  bool runOnModule(llvm::Module &M) override;

  bool isFallthrough(MetaAddress Address) const {
    return FallthroughAddresses.count(Address) != 0;
  }

  bool isFallthrough(llvm::BasicBlock *BB) const {
    return isFallthrough(getBasicBlockID(BB).start());
  }

  bool isFallthrough(llvm::Instruction *I) const {
    revng_assert(I->isTerminator());
    return isFallthrough(I->getParent());
  }

private:
  void buildFilteredCFG(llvm::Function &F);

private:
  llvm::Function *FunctionCall;
  std::set<MetaAddress> FallthroughAddresses;
  CustomCFG FilteredCFG;
};
