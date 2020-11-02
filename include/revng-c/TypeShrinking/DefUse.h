#pragma once

//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

// Standard includes
#include <unordered_map>
#include <vector>

// LLVM includes
#include "llvm/ADT/Statistic.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/ADT/GenericGraph.h"

namespace TypeShrinking {

struct DataFlowNode : public BidirectionalNode<DataFlowNode> {
  DataFlowNode(llvm::Instruction *Instruction) {
    this->Instruction = Instruction;
  }
  llvm::Instruction *Instruction;
};

GenericGraph<DataFlowNode> buildDataFlowGraph(llvm::Function &F);

class DefUse : public llvm::FunctionPass {
public:
  static char ID; // Pass identification, replacement for typeid
  DefUse() : FunctionPass(ID) {}

  bool runOnFunction(llvm::Function &F) {
    auto DataFlowGraph = buildDataFlowGraph(F);
    return false;
  }
};

} // namespace TypeShrinking
