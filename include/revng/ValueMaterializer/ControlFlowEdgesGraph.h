#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/BasicBlock.h"
#include "llvm/Support/DOTGraphTraits.h"

#include "revng/ADT/GenericGraph.h"
#include "revng/Support/Debug.h"

class ControlFlowEdgesNode {
public:
  llvm::BasicBlock *Source = nullptr;
  llvm::BasicBlock *Destination = nullptr;

public:
  ControlFlowEdgesNode(llvm::BasicBlock *BB) : Source(BB) {}
  ControlFlowEdgesNode(llvm::BasicBlock *Source,
                       llvm::BasicBlock *Destination) :
    Source(Source), Destination(Destination) {}

public:
  std::string toString() const;
};

class ControlFlowEdgesGraph
  : public GenericGraph<ForwardNode<ControlFlowEdgesNode>> {
public:
  using Node = ForwardNode<ControlFlowEdgesNode>;
  using Base = GenericGraph<Node>;
  using InstructionSet = llvm::SmallPtrSet<llvm::Instruction *, 8>;

private:
  llvm::DenseMap<llvm::BasicBlock *, Node *> NodeMap;
  InstructionSet Interesting;

public:
  ControlFlowEdgesGraph() = default;

public:
  Node *at(llvm::BasicBlock *BB) { return NodeMap[BB]; }

  const auto &interstingInstructions() const { return Interesting; }

public:
  void setInterestingInstructions(const InstructionSet &Interesting) {
    this->Interesting = Interesting;
  }

public:
  void dump() const;

public:
  static ControlFlowEdgesGraph
  fromNodeSet(const llvm::SmallPtrSetImpl<llvm::BasicBlock *> &NodeSet);
};

/// Put Source and Destination BasicBlocks in the node label
template<>
struct llvm::DOTGraphTraits<const ControlFlowEdgesGraph *>
  : public llvm::DefaultDOTGraphTraits {

  DOTGraphTraits(bool IsSimple = false) : DefaultDOTGraphTraits(IsSimple) {}

  static bool renderNodesUsingHTML() { return true; }

  static std::string getNodeLabel(const ControlFlowEdgesGraph::Node *Node,
                                  const ControlFlowEdgesGraph *Graph);

  static std::string getNodeAttributes(const ControlFlowEdgesGraph::Node *Node,
                                       const ControlFlowEdgesGraph *Graph);
};

template<>
struct llvm::DOTGraphTraits<ControlFlowEdgesGraph *>
  : public llvm::DOTGraphTraits<const ControlFlowEdgesGraph *> {};
