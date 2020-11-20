#pragma once

#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/Pass.h"

#include "revng/ADT/GenericGraph.h"

namespace TypeShrinking {

struct DataFlowNode : public BidirectionalNode<DataFlowNode> {
  DataFlowNode(llvm::Instruction *Instruction) {
    this->Instruction = Instruction;
  }
  llvm::Instruction *Instruction;
};
} // namespace TypeShrinking
