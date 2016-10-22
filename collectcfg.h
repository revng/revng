#ifndef _COLLECTCFG_H
#define _COLLECTCFG_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <fstream>
#include <map>
#include <set>

// LLVM includes
#include "llvm/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

template<typename T>
struct CompareByName {
  bool operator()(const T *LHS, const T *RHS) const {
    return LHS->getName() < RHS->getName();
  }
};

class CollectCFG : public llvm::FunctionPass {
public:
  static char ID;

public:
  CollectCFG() : llvm::FunctionPass(ID) { }

  bool runOnFunction(llvm::Function &F) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }

  void serialize(std::ostream &Output);

private:
  bool isNewInstruction(llvm::BasicBlock *BB);

private:
  std::map<llvm::BasicBlock *,
           llvm::SmallVector<llvm::BasicBlock *, 2>,
           CompareByName<llvm::BasicBlock>> Result;
  std::set<llvm::BasicBlock *> BlackList;
};

#endif // _COLLECTCFG_H
