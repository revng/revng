#ifndef REVNGC_RESTRUCTURE_CFG_UTILS_H
#define REVNGC_RESTRUCTURE_CFG_UTILS_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <cstdlib>
#include <fstream>
#include <memory>
#include <set>
#include <sys/stat.h>

// Local libraries includes
#include "revng-c/RestructureCFGPass/ASTNode.h"
#include "revng-c/RestructureCFGPass/BasicBlockNodeBB.h"

// TODO: move the definition of this object in an unique place, to avoid using
// an extern declaration
extern Logger<> CombLogger;

template<class NodeT>
using Edge = typename BasicBlockNode<NodeT>::EdgeDescriptor;

template<class NodeT>
inline void moveEdgeTarget(Edge<NodeT> Edge, BasicBlockNode<NodeT> *NewTarget) {
  auto SuccEdgeWithLabels = Edge.first->extractSuccessorEdge(Edge.second);
  SuccEdgeWithLabels.first = NewTarget;
  Edge.first->addLabeledSuccessor(SuccEdgeWithLabels);

  auto PredEdgeWithLabels = Edge.second->extractPredecessorEdge(Edge.first);
  NewTarget->addLabeledPredecessor(PredEdgeWithLabels);
}

template<class BBNodeT>
inline void addEdge(std::pair<BBNodeT *, BBNodeT *> New,
                    const typename BBNodeT::EdgeInfo &EdgeInfos) {

  New.first->addLabeledSuccessor(std::make_pair(New.second, EdgeInfos));
  New.second->addLabeledPredecessor(std::make_pair(New.first, EdgeInfos));
}

template<class BBNodeT>
inline void addPlainEdge(std::pair<BBNodeT *, BBNodeT *> New) {
  addEdge(New, typename BBNodeT::EdgeInfo());
}

template<class BBNodeT>
inline typename BBNodeT::node_edgeinfo_pair
extractLabeledEdge(std::pair<BBNodeT *, BBNodeT *> Edge) {
  Edge.second->removePredecessor(Edge.first);
  return Edge.first->extractSuccessorEdge(Edge.second);
}

template<class BBNodeT>
inline void markEdgeInlined(std::pair<BBNodeT *, BBNodeT *> Edge) {

  // TODO: Marking the edge as inlined by temporarily removing it from the graph
  //       and re-inserting it could cause problems in reordering true/false
  //       branches for conditional nodes. Consider adding a primitive for
  //       getting a reference to the `EdgeInfo` struct of an edge without
  //       having to remove it.

  // Take care of the predecessor edge.
  auto EdgePairBack = Edge.second->extractPredecessorEdge(Edge.first);
  EdgePairBack.second.Inlined = true;

  // Take care of the successor edge.
  auto EdgePairForw = Edge.first->extractSuccessorEdge(Edge.second);
  EdgePairForw.second.Inlined = true;

  // Ensure that the forward and backward edgeinfos carry the same information.
  // For this to work, we default the spaceship operator for having the equality
  // between EdgeInfo structs.
  revng_assert(EdgePairBack.second == EdgePairForw.second);

  // Re-add the removed edge.
  addEdge(Edge, EdgePairBack.second);
}

template<class NodeT>
using Stack = std::vector<std::pair<BasicBlockNode<NodeT> *, size_t>>;

template<class NodeT>
inline bool alreadyOnStack(Stack<NodeT> &Stack, BasicBlockNode<NodeT> *Node) {
  for (auto &StackElem : Stack) {
    if (StackElem.first == Node) {
      return true;
    }
  }

  return false;
}

template<class NodeT>
using BasicBlockNodeTSet = typename BasicBlockNode<NodeT>::BBNodeSet;

template<class NodeT>
inline bool alreadyOnStackQuick(BasicBlockNodeTSet<NodeT> &StackSet,
                                BasicBlockNode<NodeT> *Node) {
  if (StackSet.count(Node)) {
    return true;
  } else {
    return false;
  }
}

template<class NodeT>
// Helper function to find all nodes on paths between a source and a target
// node
inline std::set<BasicBlockNode<NodeT> *>
findReachableNodes(BasicBlockNode<NodeT> *Source,
                   BasicBlockNode<NodeT> *Target) {

  // The node starting the exploration should always exist (the same does not
  // hold for the target node).
  revng_assert(Source != nullptr);

  // Add to the Targets set the original target node.
  std::set<BasicBlockNode<NodeT> *> Targets;
  Targets.insert(Target);

  // Exploration stack initialization.
  Stack<NodeT> Stack;
  std::set<BasicBlockNode<NodeT> *> StackSet;
  Stack.push_back(std::make_pair(Source, 0));

  // Visited nodes to avoid entering in a loop.
  std::set<Edge<NodeT>> VisitedEdges;

  // Additional data structure to keep nodes that need to be added only if a
  // certain node will be added to the set of reachable nodes.
  std::map<BasicBlockNode<NodeT> *, BasicBlockNodeTSet<NodeT>> AdditionalNodes;

  // Exploration.
  while (!Stack.empty()) {
    auto StackElem = Stack.back();
    Stack.pop_back();
    BasicBlockNode<NodeT> *Vertex = StackElem.first;
    if (StackElem.second == 0) {
      if (Targets.count(Vertex) != 0) {
        for (auto StackE : Stack) {
          Targets.insert(StackE.first);
        }
        continue;
      } else if (alreadyOnStackQuick(StackSet, Vertex)) {
        // Add all the nodes on the stack to the set of additional nodes.
        BasicBlockNodeTSet<NodeT> &AdditionalSet = AdditionalNodes[Vertex];
        for (auto StackE : Stack) {
          AdditionalSet.insert(StackE.first);
        }
        continue;
      }
    }
    StackSet.insert(Vertex);

    size_t Index = StackElem.second;
    if (Index < StackElem.first->successor_size()) {
      BasicBlockNode<NodeT> *NextSuccessor = Vertex->getSuccessorI(Index);
      Index++;
      Stack.push_back(std::make_pair(Vertex, Index));
      if (VisitedEdges.count(std::make_pair(Vertex, NextSuccessor)) == 0
          and NextSuccessor != Source
          and !alreadyOnStackQuick(StackSet, NextSuccessor)) {
        Stack.push_back(std::make_pair(NextSuccessor, 0));
        VisitedEdges.insert(std::make_pair(Vertex, NextSuccessor));
      }
    } else {
      StackSet.erase(Vertex);
    }
  }

  // Add additional nodes.
  std::set<BasicBlockNode<NodeT> *> OldTargets;

  do {
    // At each iteration obtain a copy of the old set, so that we are able to
    // exit from the loop as soon no change is made to the `Targets` set.

    OldTargets = Targets;

    // Temporary storage for the nodes to add at each iteration, to avoid
    // invalidation on the `Targets` set.
    std::set<BasicBlockNode<NodeT> *> NodesToAdd;

    for (BasicBlockNode<NodeT> *Node : Targets) {
      std::set<BasicBlockNode<NodeT> *> &AdditionalSet = AdditionalNodes[Node];
      NodesToAdd.insert(AdditionalSet.begin(), AdditionalSet.end());
    }

    // Add all the additional nodes found in this step.
    Targets.insert(NodesToAdd.begin(), NodesToAdd.end());
    NodesToAdd.clear();

  } while (Targets != OldTargets);

  return Targets;
}

inline void dumpASTOnFile(std::string FolderName,
                          std::string FunctionName,
                          std::string FileName,
                          ASTNode *RootNode) {

  std::ofstream ASTFile;
  std::string PathName = FolderName + "/" + FunctionName;
  mkdir(FolderName.c_str(), 0775);
  mkdir(PathName.c_str(), 0775);
  ASTFile.open(PathName + "/" + FileName + ".dot");
  ASTFile << "digraph CFGFunction {\n";
  RootNode->dump(ASTFile);
  ASTFile << "}\n";
  ASTFile.close();
}

#endif // REVNGC_RESTRUCTURE_CFG_UTILS_H
