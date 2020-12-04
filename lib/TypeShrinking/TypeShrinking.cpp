/// \file TypeShrinking.cpp
/// \brief This analysis finds which bits of each Instruction is alive
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
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"

#include "revng-c/TypeShrinking/MFP.h"
#include "revng-c/TypeShrinking/TypeShrinking.h"

#include "BitLiveness.h"

using namespace llvm;

using BitSet = std::set<int>;
using RegisterTypeShrinking = RegisterPass<TypeShrinking::TypeShrinking>;

static cl::OptionCategory RevNgCategory("revng-c type-shrinking");

static cl::opt<uint32_t> MinimumWidth("min-width",
                                      cl::init(1),
                                      cl::desc("ignore analysis results for "
                                               "width lower than"),
                                      cl::value_desc("min-width"),
                                      cl::cat(RevNgCategory));

char TypeShrinking::TypeShrinking::ID = 0;

static RegisterTypeShrinking
  X("type-shrinking", "Run the type shrinking analysis", true, true);
namespace TypeShrinking {

/// Builds a data flow graph with edges from uses to definitions
static GenericGraph<DataFlowNode> buildDataFlowGraph(Function &F);

/// Returns true if each bit B of the result of Ins depends only on
/// the bits of the operands with an index lower than B
bool isAddLike(const Instruction *Ins) {
  switch (Ins->getOpcode()) {
  case llvm::Instruction::And:
  case llvm::Instruction::Xor:
  case llvm::Instruction::Or:
  case llvm::Instruction::Add:
  case llvm::Instruction::Sub:
  case llvm::Instruction::Mul:
    return true;
  }
  return false;
}

bool TypeShrinking::runOnFunction(Function &F) {

  auto DataFlowGraph = buildDataFlowGraph(F);
  std::vector<DataFlowNode *> ExtremalLabels;
  for (auto *Node : DataFlowGraph.nodes()) {
    if (isDataFlowSink(Node->Instruction)) {
      ExtremalLabels.push_back(Node);
    }
  }

  auto FixedPoints = getMaximalFixedPoint<BitLivenessAnalysis>(&DataFlowGraph,
                                                               0,
                                                               Top,
                                                               ExtremalLabels);
  bool HasChanges = false;

  std::vector<uint32_t> Ranks = { 8, 16, 32, 64 };

  for (auto &[Label, Result] : FixedPoints) {
    auto *Ins = Label->Instruction;
    // Find the closest rank that contains all the alive bits
    // If there is a known rank and this is an instruction that behaves like add
    // in that the least significant bits of the result depend only on the least
    // significant bits of the operands, we can down cast the operands and then
    // upcast the result
    if (Result.outValue >= MinimumWidth.getValue() && isAddLike(Ins)) {
      auto ClosestRank = std::lower_bound(Ranks.begin(),
                                          Ranks.end(),
                                          Result.outValue);

      if (ClosestRank != Ranks.end()
          && Ins->getType()->getScalarSizeInBits() > *ClosestRank) {
        auto Rank = *ClosestRank;

        HasChanges = true;
        llvm::Value *NewIns = nullptr;

        llvm::IRBuilder<> BuilderPre(Ins);
        llvm::IRBuilder<> BuilderPost(Ins->getNextNode());

        using CastOps = llvm::Instruction::CastOps;
        auto *Lhs = BuilderPre.CreateCast(CastOps::Trunc,
                                          Ins->getOperand(0),
                                          BuilderPre.getIntNTy(Rank));
        auto *Rhs = BuilderPre.CreateCast(CastOps::Trunc,
                                          Ins->getOperand(1),
                                          BuilderPre.getIntNTy(Rank));

        NewIns = BuilderPost.CreateBinOp((Instruction::BinaryOps)
                                           Ins->getOpcode(),
                                         Lhs,
                                         Rhs);
        auto *UpCasted = BuilderPost.CreateCast(Instruction::CastOps::ZExt,
                                                NewIns,
                                                Ins->getType());
        Ins->replaceAllUsesWith(UpCasted);
        Ins->eraseFromParent();
      }
    }
  }

  return HasChanges;
}

static GenericGraph<DataFlowNode> buildDataFlowGraph(Function &F) {
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
