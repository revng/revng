/// \file TopologicalOrdering.cpp

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include "llvm/ADT/PostOrderIterator.h"

#include "InternalCompute.h"

std::vector<NodeView>
extractAugmentedTopologicalOrder(InternalGraph &Graph,
                                 const LayerContainer &Layers) {
  using AugmentedNode = BidirectionalNode<NodeView>;
  using AugmentedGraph = GenericGraph<AugmentedNode>;

  // Define internal graph.
  AugmentedGraph Augmented;

  // Add original nodes in.
  std::unordered_map<size_t, AugmentedNode *> LookupTable;
  for (auto *Node : Graph.nodes()) {
    auto NewNode = Augmented.addNode(Node);
    LookupTable.emplace(Node->index(), NewNode);
  }

  // Add original edges in.
  for (auto *From : Graph.nodes())
    for (auto *To : From->successors())
      LookupTable.at(From->index())->addSuccessor(LookupTable.at(To->index()));

  // Add extra edges.
  for (size_t Layer = 0; Layer < Layers.size(); ++Layer) {
    for (size_t From = 0; From < Layers[Layer].size(); ++From) {
      for (size_t To = From + 1; To < Layers[Layer].size(); ++To) {
        auto *FromNode = LookupTable.at(Layers[Layer][From]->index());
        auto *ToNode = LookupTable.at(Layers[Layer][To]->index());
        FromNode->addSuccessor(ToNode);
      }
    }
  }

  // Ensure there's a single entry point.
  // NOTE: `EntryNode` is not added to the augmented graph and is never
  // `disconnect`ed, so be careful when using the graph from this point on.
  // The "broken" state is used as a minor optimization to avoid unnecessary
  // cleanup since the graph is stack-allocated and only exists until this
  // function is over.
  AugmentedNode EntryNode;
  for (auto *Node : Augmented.nodes())
    if (!Node->hasPredecessors())
      EntryNode.addSuccessor(Node);
  revng_assert(EntryNode.hasSuccessors());

  // Store the topological order for the augmented graph
  std::vector<NodeView> Order;
  for (auto *Node : llvm::ReversePostOrderTraversal(&EntryNode))
    if (Node != &EntryNode)
      Order.emplace_back(*Node);

  revng_assert(Order.size() == Graph.size());
  return Order;
}
