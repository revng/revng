/// \file ControlFlowEdgesGraph.cpp

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include "llvm/IR/AssemblyAnnotationWriter.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/GraphWriter.h"

#include "revng/Support/DOTGraphTraits.h"
#include "revng/Support/IRHelpers.h"
#include "revng/ValueMaterializer/ControlFlowEdgesGraph.h"

using namespace llvm;

std::string ControlFlowEdgesNode::toString() const {
  revng_assert(Source != nullptr);
  std::string Result = getName(Source);
  if (Destination != nullptr)
    Result += " -> " + getName(Destination);
  return Result;
}

void ControlFlowEdgesGraph::dump() const {
  WriteGraph(this, "cfeg");
}

using BlockSet = SmallPtrSetImpl<BasicBlock *>;

ControlFlowEdgesGraph
ControlFlowEdgesGraph::fromNodeSet(const BlockSet &NodeSet) {
  using namespace llvm;

  ControlFlowEdgesGraph Result;

  auto GetNode = [&Result](BasicBlock *BB) -> Node * {
    auto It = Result.NodeMap.find(BB);
    if (It != Result.NodeMap.end())
      return It->second;
    else
      return Result.NodeMap[BB] = Result.addNode(BB);
  };

  // Create nodes for blocks
  for (BasicBlock *BB : NodeSet) {
    // Get/create node for block
    Node *Source = GetNode(BB);

    for (BasicBlock *Destination : successors(BB)) {
      if (NodeSet.contains(Destination)) {
        // Create node representing the edge on the CFG and connect it
        Node *EdgeNode = Result.addNode(BB, Destination);
        Source->addSuccessor(EdgeNode);
        EdgeNode->addSuccessor(GetNode(Destination));
      }
    }
  }

  return Result;
}

class HighlightInstructions : public AssemblyAnnotationWriter {
public:
  using InstructionSet = SmallPtrSetImpl<Instruction *>;

private:
  const InstructionSet &Interesting;
  bool Open = false;

public:
  HighlightInstructions(const InstructionSet &Interesting) :
    Interesting(Interesting) {}

  ~HighlightInstructions() { revng_assert(not Open); }

public:
  void emitInstructionAnnot(const Instruction *I,
                            formatted_raw_ostream &Stream) final {
    if (Open) {
      Open = false;
      Stream << "</FONT>";
    }

    if (Interesting.contains(I)) {
      Open = true;
      Stream << "<FONT COLOR=\"RED\">";
    }
  }

  void finish(raw_ostream &Stream) {
    if (Open) {
      Open = false;
      Stream << "</FONT>";
    }
  }
};

using CFEG = ControlFlowEdgesGraph;

/// Put Source and Destination BasicBlocks in the node label
std::string DOTGraphTraits<const CFEG *>::getNodeLabel(const CFEG::Node *Node,
                                                       const CFEG *Graph) {
  std::string Result;

  Result += "<FONT FACE=\"monospace\">";

  if (Node->Destination != nullptr) {
    Result += "From " + getName(Node->Source) + HTMLNewline;
    Result += "  to " + getName(Node->Destination);
  } else {
    {
      HighlightInstructions Highlighter(Graph->interstingInstructions());
      raw_string_ostream Stream(Result);
      Node->Source->print(Stream, &Highlighter);
      Highlighter.finish(Stream);
    }
    replaceAll(Result, "\n", HTMLNewline);
  }

  Result += "</FONT>";

  return Result;
}

std::string
DOTGraphTraits<const CFEG *>::getNodeAttributes(const CFEG::Node *Node,
                                                const CFEG *Graph) {
  std::string Result;

  if (Node->Destination != nullptr)
    Result += "style=dashed";

  return Result;
}
