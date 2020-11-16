/// \file DefUse.cpp
/// \brief Implementation of the pass to print the DefUse edges in a readable
/// format.

//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

#include <set>
#include <sstream>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"

#include "revng-c/TypeShrinking/BitLiveness.h"
#include "revng-c/TypeShrinking/DefUse.h"
#include "revng-c/TypeShrinking/MFP.h"

using BitSet = std::set<int>;

char TypeShrinking::DefUse::ID = 0;

using RegisterDefUse = llvm::RegisterPass<TypeShrinking::DefUse>;
static RegisterDefUse X("print-def-use1", "Print DefUse edges1", true, true);

namespace TypeShrinking {

struct EndsInStoreAnalysis
  : MonotoneFramework<int, GenericGraph<DataFlowNode> *, EndsInStoreAnalysis> {
  static int combineValues(const int &lh, const int &rh) { return lh | rh; }
  static int applyTransferFunction(DataFlowNode *L, const int &E) {
    if (L->Instruction->getOpcode() == llvm::Instruction::Store) {
      return 1;
    }
    return E;
  }
  static bool isLessOrEqual(const int &lh, const int &rh) { return lh <= rh; }
};

static GenericGraph<DataFlowNode> buildDataFlowGraph(llvm::Function &F);

bool DefUse::runOnFunction(llvm::Function &F) {
  auto DataFlowGraph = buildDataFlowGraph(F);
  llvm::nodes(&DataFlowGraph);
  BitLivenessAnalysis instance;
  std::vector<DataFlowNode *> ExtremalLabels;
  for (auto *Node : DataFlowGraph.nodes()) {
    if (hasSideEffect(Node->Instruction)) {
      ExtremalLabels.push_back(Node);
    }
  }
  unsigned Max = std::numeric_limits<unsigned>::max();
  auto FixedPoints = instance.getMaximalFixedPoint(&DataFlowGraph,
                                                   0,
                                                   Max,
                                                   ExtremalLabels);
  auto hasChanges = false;
  for (auto &[Label, Result] : FixedPoints) {
    auto *Ins = Label->Instruction;
    std::stringstream AnalysisComment;
    AnalysisComment << Result.first << ' ' << Result.second;
    llvm::LLVMContext &C = Ins->getContext();
    llvm::MDNode
      *N = llvm::MDNode::get(C, llvm::MDString::get(C, AnalysisComment.str()));
    Ins->setMetadata("revng.analysis.bit-liveness", N);
    if (Ins->getOpcode() == llvm::Instruction::And
        && Ins->getType()->getScalarSizeInBits() == 64) {
      llvm::IRBuilder<> Builder(Ins->getNextNode());
      if (Result.second == 32) {
        llvm::Type *Int32 = llvm::Type::getInt32Ty(F.getContext());
        auto *DownCasted = Builder.CreateCast(llvm::Instruction::CastOps::Trunc,
                                              Ins,
                                              Int32);
        llvm::Type *Int64 = llvm::Type::getInt64Ty(F.getContext());
        auto *UpCasted = Builder.CreateCast(llvm::Instruction::CastOps::ZExt,
                                            DownCasted,
                                            Int64);
        hasChanges = true;
        for (auto &InsUse : Ins->uses()) {
          llvm::User *User = InsUse.getUser();
          if (User != UpCasted && User != DownCasted) {
            User->setOperand(InsUse.getOperandNo(), UpCasted);
          }
        }
      }
    }
  }
  return hasChanges;
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
      UseNode->addSuccessor(DefNode);
    }
  }
  return DataFlowGraph;
}
} // namespace TypeShrinking
