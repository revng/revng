/// \file AdvanedValueInfo.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Analysis/LazyValueInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Support/GraphWriter.h"

#include "revng/MFP/DOTGraphTraits.h"
#include "revng/MFP/Graph.h"
#include "revng/Support/GraphAlgorithms.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/Statistics.h"
#include "revng/ValueMaterializer/AdvancedValueInfo.h"
#include "revng/ValueMaterializer/DataFlowGraph.h"

using namespace llvm;

static Logger<> AVILogger("avi");

inline RunningStatistics AVICFEGSizeStatitistics("avi-cfeg-size");

AdvancedValueInfoMFI::LatticeElement
AdvancedValueInfoMFI::combineValues(const LatticeElement &LHS,
                                    const LatticeElement &RHS) const {
  LatticeElement Result = LHS;

  for (auto &[Key, Value] : RHS) {
    auto &ResultEntry = Result[Key];
    ResultEntry = ResultEntry.unionWith(Value);
  }

  return Result;
}

bool AdvancedValueInfoMFI::isLessOrEqual(const LatticeElement &LHS,
                                         const LatticeElement &RHS) const {
  for (const auto &[LeftEntry, RightEntry] : zipmap_range(LHS, RHS)) {
    if (LeftEntry != nullptr and RightEntry != nullptr) {
      if (not RightEntry->second.contains(LeftEntry->second)) {
        return false;
      } else {
        // All good
      }
    } else if (LeftEntry == nullptr) {
      // All good
    } else if (RightEntry == nullptr) {
      // An element is present only in the LHS
      return false;
    }
  }

  return true;
}

AdvancedValueInfoMFI::LatticeElement
AdvancedValueInfoMFI::applyTransferFunction(Label L,
                                            const LatticeElement &E) const {

  revng_log(AVILogger, "Processing node " << L->toString());
  LoggerIndent<> Indent(AVILogger);

  LatticeElement Result = E;

  for (Instruction *I : Instructions) {
    ConstantRangeSet Range;

    if (I->getParent() != L->Source and not DT.dominates(I, L->Source)) {
      revng_log(AVILogger,
                "Skipping " << getName(I) << ": not dominated by "
                            << getName(L->Source));
      continue;
    }

    if (L->Destination != nullptr) {
      Range = LVI.getConstantRangeOnEdge(I, L->Source, L->Destination, Context);
    } else {
      Range = LVI.getConstantRange(I, L->Source->getTerminator());
    }

    if (AVILogger.isEnabled()) {
      AVILogger << "Range for " << getName(I) << ": ";
      Range.dump(AVILogger);
      AVILogger << DoLog;
    }

    auto It = Result.find(I);
    if (It != Result.end())
      It->second = It->second.intersectWith(Range);
    else
      Result[I] = Range;
  }

  return Result;
}

void AdvancedValueInfoMFI::dump(GraphType CFEG, const ResultsMap &AllResults) {
  MFP::Graph<AdvancedValueInfoMFI> MFPGraph(CFEG, AllResults);
  llvm::WriteGraph(&MFPGraph, "cfeg");
}

/// \p DFG the data flow graph containing the instructions we're interested in.
/// \p Context the position in the function for the current query.
std::tuple<std::map<llvm::Instruction *, ConstantRangeSet>,
           ControlFlowEdgesGraph,
           map<const ForwardNode<ControlFlowEdgesNode> *,
               MFP::MFPResult<map<llvm::Instruction *, ConstantRangeSet>>>>
runAVI(const DataFlowGraph &DFG,
       llvm::Instruction *Context,
       const llvm::DominatorTree &DT,
       llvm::LazyValueInfo &LVI) {
  using namespace llvm;

  //
  // Identify nodes from the root of the DFG to all the instructions in the DFG
  //

  SmallPtrSet<BasicBlock *, 4> Whitelist;

  BasicBlock *ContextBB = Context->getParent();
  SmallPtrSet<llvm::Instruction *, 8> Targets;

  // Identify all the instructions in the DFG
  for (const DataFlowGraph::Node *Node : post_order(&DFG)) {
    Value *Value = Node->Value;

    // LVI can only work on integer types
    if (not Value->getType()->isIntegerTy())
      continue;

    if (auto *I = dyn_cast<Instruction>(Value)) {
      // Register instruction to be tracked
      Targets.insert(I);
    }
  }

  if (Targets.size() == 0) {
    return {
      std::map<llvm::Instruction *, ConstantRangeSet>{},
      ControlFlowEdgesGraph(),
      map<const ForwardNode<ControlFlowEdgesNode> *,
          MFP::MFPResult<map<llvm::Instruction *, ConstantRangeSet>>>{}
    };
  }

  //
  // Collect nodes from which we should start the exploration
  //
  SmallPtrSet<BasicBlock *, 4> DFGEntryPoints;

  // Collect in a set all the blocks of the target instructions
  SmallPtrSet<BasicBlock *, 16> InstructionBlocks;
  for (Instruction *I : Targets) {
    auto *BB = I->getParent();
    InstructionBlocks.insert(BB);
  }

  SmallPtrSet<BasicBlock *, 16> NodeSet;
  for (BasicBlock *BB : InstructionBlocks) {

    // Is the block of the current instruction already whitelisted?
    if (NodeSet.contains(BB))
      continue;

    DFGEntryPoints.insert(BB);

    // Taint all the nodes to go from ContextBB to LimitedStartBB
    auto Nodes = nodesBetweenReverse(ContextBB, BB);

    NodeSet.insert(Nodes.begin(), Nodes.end());
  }

  NodeSet.insert(ContextBB);

  //
  // Create the subgraph of the CFG containing Whitelist
  //
  auto CFEG = ControlFlowEdgesGraph::fromNodeSet(NodeSet);
  CFEG.setInterestingInstructions(Targets);

  revng_log(ValueMaterializerLogger,
            "The CFG subset we're interested in has " << CFEG.size()
                                                      << " nodes");
  AVICFEGSizeStatitistics.push(CFEG.size());

  //
  // Run the MFP
  //

  // Identify initial nodes for the monotone framework
  std::vector<const ControlFlowEdgesGraph::Node *> InitialNodes;
  for (BasicBlock *Entry : DFGEntryPoints)
    InitialNodes.push_back(CFEG.at(Entry));

  revng_assert(InitialNodes.size() > 0);

  // Run MFP
  AdvancedValueInfoMFI AVIMFI(LVI, DT, Context, Targets);
  auto AllResults = MFP::getMaximalFixedPoint(AVIMFI,
                                              &CFEG,
                                              {},
                                              {},
                                              {},
                                              InitialNodes);

  if (AVILogger.isEnabled()) {
    for (const auto &[Node, AnalysisResults] : AllResults) {
      auto Dump =
        [&](const std::map<llvm::Instruction *, ConstantRangeSet> &Map) {
          for (const auto &[I, Range] : Map) {
            AVILogger << "    " << getName(I) << ": ";
            Range.dump(AVILogger);
            AVILogger << "\n";
          }
        };

      AVILogger << Node->toString() << ":\n";
      AVILogger << "  Initial value:\n";
      MFP::dump(*AVILogger.getAsLLVMStream().get(), 2, AnalysisResults.InValue);
      AVILogger << "  Final value:\n";
      MFP::dump(*AVILogger.getAsLLVMStream().get(),
                2,
                AnalysisResults.OutValue);
    }
    AVILogger << DoLog;
  }

  auto &ResultsOnTarget = AllResults.at(CFEG.at(ContextBB)).OutValue;

  return { ResultsOnTarget, std::move(CFEG), std::move(AllResults) };
}

template<>
void MFP::dump(llvm::raw_ostream &Stream,
               unsigned Indent,
               const std::map<llvm::Instruction *, ConstantRangeSet> &Element) {
  for (const auto &[I, Range] : Element) {
    for (unsigned I = 0; I < Indent; ++I)
      Stream << "  ";
    Stream << getName(I) << ": ";
    Range.dump(Stream, aviFormatter);
    Stream << "\n";
  }
}
