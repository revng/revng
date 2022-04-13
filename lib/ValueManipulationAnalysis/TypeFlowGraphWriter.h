#pragma once

//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include <map>
#include <ostream>

#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/Support/DOTGraphTraits.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/GraphWriter.h"

#include "revng/Support/Assert.h"
#include "revng/Support/IRHelpers.h"

#include "revng-c/ValueManipulationAnalysis/TypeColors.h"

#include "TypeFlowGraph.h"

namespace llvm {

template<>
struct DOTGraphTraits<vma::TypeFlowGraph *> : public DefaultDOTGraphTraits {
  using GraphT = vma::TypeFlowGraph *;
  using NodeT = typename llvm::GraphTraits<GraphT>::NodeRef;
  using EdgeIteratorT = typename llvm::GraphTraits<GraphT>::ChildIteratorType;
  static constexpr inline int IgnoreGraphvizPorts = -1;

  // Colors for printing ColorSets
  static constexpr llvm::StringRef ColorCode[vma::MAX_COLORS] = {
    // pointerness
    "\"#8db596\"",
    // unsignedness
    "\"#ec5858\"",
    // boolness
    "\"#93abd3\"",
    // signedness
    "\"#fd8c04\"",
    // floatness
    "\"#734046\"",
    // numberness
    "white"
  };

  DOTGraphTraits(bool IsSimple = false) : DefaultDOTGraphTraits(IsSimple) {}

  static std::string getGraphName(GraphT G) {
    StringRef FuncName = G->Func->getName();
    return "TypeFlowGraph of " + FuncName.str();
  }

  static std::string getNodeLabel(NodeT Node, GraphT Graph) {
    std::string OutSStr;
    llvm::raw_string_ostream Out(OutSStr);

    // Node label
    if (Node->isValue()) {
      const Value *V = Node->getValue();

      if (isa<Argument>(V))
        Out << "arg ";

      if (isa<Instruction>(V))
        Out << *(dyn_cast<Instruction>(V));
      else
        V->printAsOperand(Out);

      Out << "}\n{";
    } else {
      const Use *U = Node->getUse();

      // Const uses have labels, since const nodes are not printed
      if (isa<ConstantData>(U->get())) {
        Out << "const ";
        U->get()->printAsOperand(Out);

        Out << "}\n{";
      }
    }

    // Node colors
    Out << "color : " << dumpToString(Node->Candidates)
        << "}\n{accepted: " << dumpToString(Node->Accepted);

    return Out.str();
  }

  static std::string getNodeAttributes(NodeT Node, GraphT Graph) {
    std::string OutSStr;
    llvm::raw_string_ostream Out(OutSStr);

    // Draw a square for values and a circle for uses
    if (Node->isUse()) {
      const Use *U = Node->getUse();

      Out << "shape=oval, width=0.3, height=0.3, tooltip=\"operand #"
          << U->getOperandNo() << "\"";
    } else {
      Out << "shape=box ";
      if (Node->isValue()
          and (Node->getValue()->getType()->isLabelTy()
               or Node->getValue()->getType()->isVoidTy()))
        Out << ", color=lightgrey ";
    }

    // Fill nodes: 0 colors = white, 1 color = colored, > 1 color = grey
    if (Node->isUndecided()) {
      Out << " style=filled, fillcolor=lightgrey, color=lightgrey";
    } else if (Node->isDecided()) {
      auto First = Node->Candidates.firstSetBit();
      revng_assert(First != vma::NUMBERNESS_INDEX);

      Out << " style=filled, fillcolor=" << ColorCode[First].str()
          << ", color=" << ColorCode[First].str();
    }

    return Out.str();
  }

  static bool isNodeHidden(NodeT Node, const GraphT &) {
    // Don't print a node and its incoming/outgoing edges if it's a constant
    // Constant values are printed in their Uses nodes, to not pollute the
    // graph with meaningless arcs between the same constant and multiple
    // uses.
    if (Node->isValue() and isa<ConstantData>(Node->getValue()))
      return true;
    return false;
  }

  static std::string
  getEdgeAttributes(NodeT Node, EdgeIteratorT EI, GraphT Graph) {
    return "constraint=false, color=\"#8bcdcd\", fontcolor=\"#3797a4\", "
           "labeldistance=2,"
           "labelangle=40,"
           " headlabel=\""
           + dumpToString(EI.getCurrent()->Colors) + "\"";
  }

  static void addCustomGraphFeatures(GraphT G, GraphWriter<GraphT> &GW) {
    const Function *F = G->Func;

    // Add dataflow arrows (Operand -> OpUse -> Instruction):
    // This gives a better ranking of the nodes and a better overall
    // understanding of the graph.
    for (const Instruction &I : instructions(*F)) {
      if (not G->ContentToNodeMap.contains(&I)
          or isNodeHidden(G->getNodeContaining(&I), G))
        continue;

      NodeT ValNode = G->getNodeContaining(&I);

      for (const auto &O : I.operands()) {
        // Connect Operand Uses to Instructions
        if (not G->ContentToNodeMap.contains(&O)
            or isNodeHidden(G->getNodeContaining(&O), G))
          continue;

        NodeT OpUse = G->getNodeContaining(&O);
        GW.emitEdge(OpUse,
                    IgnoreGraphvizPorts,
                    ValNode,
                    IgnoreGraphvizPorts,
                    "color=lightgrey, minlen=2");

        // Connect Operands to Operand Uses
        if (not G->ContentToNodeMap.contains(O.get())
            or isNodeHidden(G->getNodeContaining(O.get()), G))
          continue;

        NodeT OpNode = G->getNodeContaining(O.get());
        GW.emitEdge(OpNode,
                    IgnoreGraphvizPorts,
                    OpUse,
                    IgnoreGraphvizPorts,
                    "color=lightgrey, minlen=2");
      }
    }
  }
};

} // namespace llvm
