/// \file TypeShrinking.cpp
/// \brief This analysis finds which bits of each llvm::Instruction is alive
/// format.

//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

#include <set>
#include <sstream>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Casting.h"

#include "revng-c/TypeShrinking/MFP.h"
#include "revng-c/TypeShrinking/TypeShrinking.h"

#include "BitLiveness.h"

using BitSet = std::set<int>;

char TypeShrinking::TypeShrinking::ID = 0;

using RegisterTypeShrinking = llvm::RegisterPass<TypeShrinking::TypeShrinking>;
static RegisterTypeShrinking
  X("type-shrinking", "Run the type shrinking analysis", true, true);

namespace TypeShrinking {

/// Builds a data flow graph with edges from uses to definitions
static GenericGraph<DataFlowNode> buildDataFlowGraph(llvm::Function &F);

bool TypeShrinking::runOnFunction(llvm::Function &F) {
  auto DataFlowGraph = buildDataFlowGraph(F);
  llvm::nodes(&DataFlowGraph);
  std::vector<DataFlowNode *> ExtremalLabels;
  for (auto *Node : DataFlowGraph.nodes()) {
    if (hasSideEffect(Node->Instruction)) {
      ExtremalLabels.push_back(Node);
    }
  }

  unsigned Max = std::numeric_limits<unsigned>::max();
  auto FixedPoints = BitLivenessAnalysis::getMaximalFixedPoint(&DataFlowGraph,
                                                               0,
                                                               Max,
                                                               ExtremalLabels);
  bool hasChanges = false;

  std::vector<unsigned> Ranks = { 8, 16, 32, 64 };
  const auto IsAddLike = [](unsigned OpCode) {
    switch (OpCode) {
    case llvm::Instruction::And:
    case llvm::Instruction::Xor:
    case llvm::Instruction::Or:
    case llvm::Instruction::Add:
    case llvm::Instruction::Sub:
    case llvm::Instruction::Mul:
      return true;
    }
    return false;
  };

  for (auto &[Label, Result] : FixedPoints) {
    auto *Ins = Label->Instruction;
    // Find the closest rank that contains all the alive bits
    auto ClosestRank = std::lower_bound(Ranks.begin(),
                                        Ranks.end(),
                                        Result.second);
    // If there is a known rank and this is an instruction that behaves like add
    // in that the least significant bits of the result depend only on the least
    // significant bits of the operands, we can down cast the operands and then
    // upcast the result
    if (ClosestRank != Ranks.end() && IsAddLike(Ins->getOpcode())) {
      auto Rank = *ClosestRank;
      if (Ins->getType()->getScalarSizeInBits() > Rank) {
        hasChanges = true;
        llvm::IRBuilder<> BuilderPre(Ins);
        llvm::IRBuilder<> BuilderPost(Ins->getNextNode());
        llvm::Value *NewIns = nullptr;
        using CastOps = llvm::Instruction::CastOps;
        auto *Lhs = BuilderPre.CreateCast(CastOps::Trunc,
                                          Ins->getOperand(0),
                                          llvm::Type::getIntNTy(F.getContext(),
                                                                Rank));
        auto *Rhs = BuilderPre.CreateCast(CastOps::Trunc,
                                          Ins->getOperand(1),
                                          llvm::Type::getIntNTy(F.getContext(),
                                                                Rank));

        NewIns = BuilderPost.CreateBinOp((llvm::Instruction::BinaryOps)
                                           Ins->getOpcode(),
                                         Lhs,
                                         Rhs);
        auto *UpCasted = BuilderPost.CreateCast(CastOps::ZExt,
                                                NewIns,
                                                Ins->getType());
        Ins->replaceAllUsesWith(UpCasted);
        Ins->eraseFromParent();
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
