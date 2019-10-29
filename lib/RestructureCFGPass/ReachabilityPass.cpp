/// \file ReachabilityPass.cpp
/// \brief FunctionPass that computes the reachability for the nodes of a given
///        Function

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <sstream>
#include <stdlib.h>

// LLVM includes
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

// revng includes
#include "revng/Support/Debug.h"
#include "revng/Support/IRHelpers.h"

// Local libraries includes
#include "revng-c/RestructureCFGPass/ReachabilityPass.h"

using namespace llvm;

char ReachabilityPass::ID = 0;
using Reg = RegisterPass<ReachabilityPass>;
static Reg X("reachability", "Compute reachability information", true, true);

bool ReachabilityPass::runOnFunction(Function &F) {

  // Clean class members.
  ReachableBlocks.clear();

  std::map<BasicBlock *, unsigned> BBToIndex;
  std::map<unsigned, BasicBlock *> IndexToBB;
  unsigned Index = 0;

  // Initialize a mapping between basic blocks and their index.
  for (BasicBlock &BB : F) {
    BBToIndex[&BB] = Index;
    IndexToBB[Index] = &BB;
    Index++;
  }

  // Maximum index of the basic blocks.
  unsigned Dimension = Index;
  unsigned MaxIndex = Index - 1;

  // Create and initialize the incidence matrix.
  using Bool2DMatrix = std::vector<std::vector<bool>>;
  Bool2DMatrix Matrix(Dimension, std::vector<bool>(Dimension, false));
  for (unsigned i = 0; i <= MaxIndex; i++) {
    for (unsigned j = 0; j <= MaxIndex; j++) {
      if (i == j) {
        Matrix[i][j] = 1;
      } else {
        Matrix[i][j] = 0;
      }
    }
  }

  // Fill the incidence matrix with the connections at single step.
  for (BasicBlock &BB : F) {
    unsigned BBIndex = BBToIndex[&BB];
    TerminatorInst *Terminator = BB.getTerminator();
    for (BasicBlock *Successor : Terminator->successors()) {
      unsigned SuccessorIndex = BBToIndex[Successor];
      Matrix[BBIndex][SuccessorIndex] = 1;
    }
  }

  dbg << "Mapping:\n";
  for (auto &Elem : BBToIndex) {
    dbg << getName(Elem.first) << " " << Elem.second << "\n";
  }

  dbg << "Matrix is:\n";
  for (unsigned i = 0; i <= MaxIndex; i++) {
    for (unsigned j = 0; j <= MaxIndex; j++) {
      dbg << Matrix[i][j] << " ";
    }
    dbg << "\n";
  }

  bool Change = true;
  while (Change) {
    Change = false;

  Bool2DMatrix MatrixClosure(Dimension, std::vector<bool>(Dimension, false));
    for (unsigned i = 0; i <= MaxIndex; i++) {
      for (unsigned j = 0; j <= MaxIndex; j++) {
        bool Value = 0;
        for (unsigned k = 0; k <= MaxIndex; k++) {
        }
        MatrixClosure[i][j] = Value;
      }
    }

    dbg << "Matrix closure is:\n";
    for (unsigned i = 0; i <= MaxIndex; i++) {
      for (unsigned j = 0; j <= MaxIndex; j++) {
        dbg << MatrixClosure[i][j] << " ";
      }
      dbg << "\n";
    }

    for (unsigned i = 0; i <= MaxIndex; i++) {
      for (unsigned j = 0; j <= MaxIndex; j++) {
        bool OldValue = Matrix[i][j];
        Matrix[i][j] = Matrix[i][j] or MatrixClosure[i][j];
        bool NewValue = Matrix[i][j];
        if (OldValue != NewValue) {
          Change = true;
        }
      }
    }

    dbg << "Matrix sum is:\n";
    for (unsigned i = 0; i <= MaxIndex; i++) {
      for (unsigned j = 0; j <= MaxIndex; j++) {
        dbg << Matrix[i][j] << " ";
      }
      dbg << "\n";
    }
  }

  // Fill the final data structure.
  for (unsigned i = 0; i <= MaxIndex; i++) {
    for (unsigned j = 0; j <= MaxIndex; j++) {
      BasicBlock *SourceBB = IndexToBB[i];
      BasicBlock *TargetBB = IndexToBB[j];
      if (Matrix[i][j]) {
        ReachableBlocks[SourceBB].insert(TargetBB);
      }
    }
  }

  // Print the final data structure.
  for (auto &It : ReachableBlocks) {
    dbg << "From " << getName(It.first) << " I can reach:\n";
    for (auto Elem : It.second) {
      dbg << getName(Elem) << "\n";
    }
  }

  return false;
}

bool ReachabilityPass::existsPath(BasicBlock *Source, BasicBlock *Target) {
  if (ReachableBlocks[Source].count(Target) != 0) {
    return true;
  } else {
    return false;
  }
}

std::set<BasicBlock *> &ReachabilityPass::reachableFrom(BasicBlock *Source) {
  return ReachableBlocks[Source];
}
