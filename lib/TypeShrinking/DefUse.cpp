/// \file DefUse.cpp
/// \brief Implementation of the pass to print the DefUse edges in a readable
/// format.

//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

#include <set>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "llvm/IR/InstIterator.h"

#include "revng-c/TypeShrinking/DefUse.h"
#include "revng-c/TypeShrinking/MFP.h"

using BitSet = std::set<int>;

char TypeShrinking::DefUse::ID = 0;

using RegisterDefUse = llvm::RegisterPass<TypeShrinking::DefUse>;
static RegisterDefUse X("print-def-use1", "Print DefUse edges1", true, true);

namespace TypeShrinking {

struct DataFlowNode : public BidirectionalNode<DataFlowNode> {
  DataFlowNode(llvm::Instruction *Instruction) {
    this->Instruction = Instruction;
  }
  llvm::Instruction *Instruction;
};

static GenericGraph<DataFlowNode> buildDataFlowGraph(llvm::Function &F);

bool DefUse::runOnFunction(llvm::Function &F) {
  auto DataFlowGraph = buildDataFlowGraph(F);
  return false;
}

static GenericGraph<DataFlowNode> buildDataFlowGraph(llvm::Function &F) {
  GenericGraph<DataFlowNode> DataFlowGraph{};
  std::vector<DataFlowNode *> Worklist;
  std::unordered_map<llvm::Instruction *, DataFlowNode *> InstructionNodeMap;
  for (auto I = inst_begin(F), E = inst_end(F); I != E; ++I) {
    DataFlowNode Node{ &*I };
    auto *GraphNode = DataFlowGraph.addNode(Node);
    Worklist.push_back(GraphNode);
    InstructionNodeMap[GraphNode->Instruction] = GraphNode;
  }

  for (auto *DefNode : Worklist) {
    auto *Ins = DefNode->Instruction;
    for (auto &Use : llvm::make_range(Ins->use_begin(), Ins->use_end())) {
      auto *UserInstr = llvm::cast<llvm::Instruction>(Use.getUser());
      auto *UseNode = InstructionNodeMap[UserInstr];
      llvm::errs() << "Found use " << DefNode << ' ' << UseNode << '\n';
      DefNode->addSuccessor(UseNode);
    }
  }
  return DataFlowGraph;
}
} // namespace TypeShrinking
