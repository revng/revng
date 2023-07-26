#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/Pass.h"
#include "llvm/Support/DOTGraphTraits.h"

#include "revng/ADT/GenericGraph.h"

namespace TypeShrinking {

struct DataFlowNodeData {
  DataFlowNodeData(llvm::Instruction *Ins) : Instruction(Ins){};
  llvm::Instruction *Instruction;
};

using DataFlowNode = BidirectionalNode<DataFlowNodeData>;
using DataFlowGraph = GenericGraph<DataFlowNode>;

/// Builds a data flow graph with edges from uses to definitions
DataFlowGraph buildDataFlowGraph(llvm::Function &F);

} // namespace TypeShrinking

template<>
struct llvm::DOTGraphTraits<const TypeShrinking::DataFlowGraph *>
  : public llvm::DefaultDOTGraphTraits {

  DOTGraphTraits(bool IsSimple = false) : DefaultDOTGraphTraits(IsSimple) {}

  static std::string getGraphProperties(const TypeShrinking::DataFlowGraph *);

  static std::string getNodeLabel(const TypeShrinking::DataFlowNode *Node,
                                  const TypeShrinking::DataFlowGraph *Graph);

  static std::string
  getNodeAttributes(const TypeShrinking::DataFlowNode *Node,
                    const TypeShrinking::DataFlowGraph *Graph) {
    return "";
  }
};

template<>
struct llvm::DOTGraphTraits<TypeShrinking::DataFlowGraph *>
  : public llvm::DOTGraphTraits<const TypeShrinking::DataFlowGraph *> {};
