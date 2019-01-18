#ifndef COLLECTCFG_H
#define COLLECTCFG_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <fstream>
#include <map>
#include <set>

// LLVM includes
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Pass.h"

template<typename T>
struct CompareByName {
  bool operator()(const T *LHS, const T *RHS) const {
    return LHS->getName() < RHS->getName();
  }
};

class CollectCFG : public llvm::ModulePass {
public:
  static char ID;

public:
  CollectCFG() : llvm::ModulePass(ID) {}

  bool runOnModule(llvm::Module &M) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }

  void serialize(std::ostream &Output);

private:
  bool isNewInstruction(llvm::BasicBlock *BB);

private:
  using BasicBlock = llvm::BasicBlock;

  template<typename T, size_t N>
  using SmallVector = llvm::SmallVector<T, N>;

  using Comparer = CompareByName<BasicBlock>;
  std::map<BasicBlock *, SmallVector<BasicBlock *, 2>, Comparer> Result;
  std::set<BasicBlock *> BlackList;
};

#endif // COLLECTCFG_H
