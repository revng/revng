/// \file DataFlowGraph.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <unordered_map>

#include "llvm/IR/InstIterator.h"

#include "revng/TypeShrinking/DataFlowGraph.h"

using namespace llvm;

namespace TypeShrinking {
GenericGraph<DataFlowNode> buildDataFlowGraph(Function &F) {
  GenericGraph<DataFlowNode> DataFlowGraph{};
  std::vector<DataFlowNode *> Worklist;
  std::unordered_map<Instruction *, DataFlowNode *> InstructionNodeMap;
  // Initialization
  for (Instruction &I : instructions(F)) {
    DataFlowNode Node{ &I };
    auto *GraphNode = DataFlowGraph.addNode(Node);
    Worklist.push_back(GraphNode);
    InstructionNodeMap[GraphNode->Instruction] = GraphNode;
  }

  for (auto *DefNode : Worklist) {
    auto *Ins = DefNode->Instruction;
    for (auto &Use : Ins->uses()) {
      auto *UseNode = InstructionNodeMap.at(cast<Instruction>(Use.getUser()));
      UseNode->addSuccessor(DefNode);
    }
  }
  return DataFlowGraph;
}

} // namespace TypeShrinking
