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

namespace TypeShrinking {

template std::unordered_map<DataFlowNode *,
                            std::tuple<std::set<int>, std::set<int>>>
getMaximalFixedPoint(std::set<int> (*combineValues)(std::set<int> &,
                                                    std::set<int> &),
                     bool (*isLess)(std::set<int> &, std::set<int> &),
                     bool (*areEqual)(std::set<int> &, std::set<int> &),
                     GenericGraph<DataFlowNode> &Flow,
                     std::set<int> ExtremalValue,
                     std::set<int> BottomValue,
                     std::vector<DataFlowNode *> &ExtremalLabels,
                     // Temporarily disable clang-format here. It conflicts with
                     // revng conventions
                     // clang-format off
                     std::function<std::set<int>(std::set<int> &)>
                       (*getTransferFunction)(DataFlowNode *)
                     // clang-format on
);

GenericGraph<DataFlowNode> buildDataFlowGraph(llvm::Function &F) {
  GenericGraph<DataFlowNode> DataFlowGraph{};
  std::vector<DataFlowNode *> Worklist;
  std::unordered_map<llvm::Instruction *, DataFlowNode *> InstructionNodeMap;
  for (llvm::inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I) {
    DataFlowNode Node{ &*I };
    auto GraphNode = DataFlowGraph.addNode(Node);
    Worklist.push_back(GraphNode);
    InstructionNodeMap[GraphNode->Instruction] = GraphNode;
  }

  /// def-use chain for llvm::Instruction
  // it's better to use this thingie and then reverse the edges
  for (auto I = Worklist.begin(), E = Worklist.end(); I != E; ++I) {
    auto *DefNode = *I;
    auto *Ins = DefNode->Instruction;
    for (auto &Use : llvm::make_range(Ins->use_begin(), Ins->use_end())) {
      auto *UserInstr = llvm::dyn_cast<llvm::Instruction>(Use.getUser());
      auto *UseNode = InstructionNodeMap[UserInstr];
      llvm::errs() << "Found use " << DefNode << ' ' << UseNode << '\n';
      DefNode->addSuccessor(UseNode);
    }
  }
  return DataFlowGraph;
}
} // namespace TypeShrinking

using namespace llvm;

char TypeShrinking::DefUse::ID = 0;
using RegisterDefUse = RegisterPass<TypeShrinking::DefUse>;
static RegisterDefUse X("print-def-use1", "Print DefUse edges1", true, true);
