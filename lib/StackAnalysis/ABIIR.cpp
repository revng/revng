/// \file ABIIR.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// LLVM includes
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/PostOrderIterator.h"

// Local includes
#include "ABIIR.h"

namespace StackAnalysis {

void ABIFunction::finalize() {
  revng_assert(Calls.empty());

  for (auto &P : BBMap) {
    // Build backward links
    for (ABIIRBasicBlock *Successor : P.second.Successors)
      Successor->Predecessors.push_back(&P.second);

    if (P.second.successor_size() == 0)
      FinalBBs.push_back(&P.second);

    // Find all the function calls
    for (ABIIRInstruction &I : P.second)
      if (I.isCall())
        Calls.emplace_back(&P.second, &I);
  }

  // The entry point should not have predecessors
  if (IREntry->predecessor_size() != 0) {
    ABIIRBasicBlock *NewEntry = &this->get(nullptr);
    NewEntry->addSuccessor(IREntry);
    IREntry->Predecessors.push_back(NewEntry);
    IREntry = NewEntry;
  }
}

bool ABIFunction::verify() const {
  for (auto &P : BBMap) {
    const ABIIRBasicBlock &BB = P.second;

    for (auto &Successor : BB.successors()) {
      auto SuccessorPredecessors = Successor->predecessors();
      auto StartIt = SuccessorPredecessors.begin();
      auto EndIt = SuccessorPredecessors.end();
      if (std::find(StartIt, EndIt, &BB) == EndIt)
        return false;
    }

    for (auto &Predecessor : BB.predecessors()) {
      auto PredecessorSuccessors = Predecessor->successors();
      auto StartIt = PredecessorSuccessors.begin();
      auto EndIt = PredecessorSuccessors.end();
      if (std::find(StartIt, EndIt, &BB) == EndIt)
        return false;
    }

    if (BB.predecessor_size() == 0 and &BB != IREntry)
      return false;
  }

  return true;
}

std::set<int32_t> ABIFunction::writtenRegisters() const {
  std::set<int32_t> WrittenRegisters;

  for (const auto &P : BBMap)
    for (const ABIIRInstruction &I : P.second)
      if (I.isStore() and I.target().addressSpace() == ASID::cpuID())
        WrittenRegisters.insert(I.target().offset());

  return WrittenRegisters;
}

std::set<FunctionCall> ABIFunction::incoherentCalls() {
  std::vector<ABIIRBasicBlock *> Extremals;
  for (auto &P : BBMap)
    if (P.second.successor_size() == 0)
      Extremals.push_back(&P.second);

  return computeIncoherentCalls(entry(), Extremals);
}

void ABIFunction::dumpDot() const {
  std::map<const ABIIRBasicBlock *, unsigned> RPOTPosition;
  {
    llvm::ReversePostOrderTraversal<ABIIRBasicBlock *> RPOT(IREntry);
    unsigned I = 0;
    for (ABIIRBasicBlock *BB : RPOT)
      RPOTPosition[BB] = I++;
  }

  dbg << "digraph ABIFunction {\n";

  for (auto &P : BBMap) {
    const ABIIRBasicBlock &BB = P.second;
    dbg << "\"" << getName(BB.basicBlock()) << "\" [";
    dbg << "label=\"" << getName(BB.basicBlock()) << " ";

    auto It = RPOTPosition.find(&BB);
    if (It == RPOTPosition.end())
      dbg << "N/A";
    else
      dbg << It->second;

    dbg << "\"";
    if (&BB == IREntry)
      dbg << "fillcolor=green,style=filled";
    dbg << "];\n";

    for (auto &Successor : BB.successors()) {
      dbg << "\"" << getName(BB.basicBlock()) << "\""
          << " -> \"" << getName(Successor->basicBlock()) << "\""
          << " [color=green];\n";
    }
  }

  dbg << "}\n";
}

} // namespace StackAnalysis
