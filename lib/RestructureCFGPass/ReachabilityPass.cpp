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

// Local libraries includes
#include "ReachabilityPass.h"
#include "revng/Support/Debug.h"
#include "revng/Support/IRHelpers.h"

using namespace llvm;

char ReachabilityPass::ID = 0;
static RegisterPass<ReachabilityPass> X("reachability",
                                        "Compute reachability information",
                                        true,
                                        true);

bool ReachabilityPass::runOnFunction(Function &F) {

  // Clean class members.
  ReachableBlocks.clear();

  std::map<BasicBlock *, int> BBToIndex;
  std::map<int, BasicBlock *> IndexToBB;
  int Index = 0;

  // Initialize a mapping between basic blocks and their index.
  for (BasicBlock &BB : F) {
    BBToIndex[&BB] = Index;
    IndexToBB[Index] = &BB;
    Index++;
  }

  // Maximum index of the basic blocks.
  int Dimension = Index;
  int MaxIndex = Index - 1;

  // Create and initialize the incidence matrix.
  bool Matrix [Dimension][Dimension];
  for (int i=0; i<=MaxIndex; i++) {
    for (int j=0; j<=MaxIndex; j++) {
      if (i == j) {
        Matrix[i][j] = 1;
      } else {
        Matrix[i][j] = 0;
      }
    }
  }

  // Fill the incidence matrix with the connections at single step.
  for (BasicBlock &BB : F) {
    int BBIndex = BBToIndex[&BB];
    TerminatorInst *Terminator = BB.getTerminator();
    for (BasicBlock *Successor : Terminator->successors()) {
      int SuccessorIndex = BBToIndex[Successor];
      Matrix[BBIndex][SuccessorIndex] = 1;
    }
  }

  dbg << "Mapping:\n";
  for (auto &Elem : BBToIndex) {
    dbg << getName(Elem.first) << " " << Elem.second << "\n";
  }

  dbg << "Matrix is:\n";
  for (int i=0; i<=MaxIndex; i++) {
    for (int j=0; j<=MaxIndex; j++) {
      dbg << Matrix[i][j] << " ";
    }
    dbg << "\n";
  }


  bool Change = true;
  while (Change) {
    Change = false;

    bool MatrixClosure [Dimension][Dimension];
    for (int i=0; i<=MaxIndex; i++) {
      for (int j=0; j<=MaxIndex; j++) {
        bool Value = 0;
        for (int k=0; k<=MaxIndex; k++) {
          Value = Value or (Matrix[i][k] and Matrix[k][j]);
        }
        MatrixClosure[i][j] = Value;
      }
    }

    dbg << "Matrix closure is:\n";
    for (int i=0; i<=MaxIndex; i++) {
      for (int j=0; j<=MaxIndex; j++) {
        dbg << MatrixClosure[i][j] << " ";
      }
      dbg << "\n";
    }

    for (int i=0; i<=MaxIndex; i++) {
      for (int j=0; j<=MaxIndex; j++) {
        bool OldValue = Matrix[i][j];
        Matrix[i][j] |= MatrixClosure[i][j];
        bool NewValue = Matrix[i][j];
        if (OldValue != NewValue) {
          Change = true;
        }
      }
    }

    dbg << "Matrix sum is:\n";
    for (int i=0; i<=MaxIndex; i++) {
      for (int j=0; j<=MaxIndex; j++) {
        dbg << Matrix[i][j] << " ";
      }
      dbg << "\n";
    }
  }

  // Fill the final data structure.
  for (int i=0; i<=MaxIndex; i++) {
    for (int j=0; j<=MaxIndex; j++) {
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
