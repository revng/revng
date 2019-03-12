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
#include "revng-c/RestructureCFGPass/BasicBlockNode.h"

// TODO: move the definition of this object in an unique place, to avoid using
// an extern declaration
extern Logger<> CombLogger;

// EdgeDescriptor is a handy way to create and manipulate edges on the
// RegionCFG.
using EdgeDescriptor = std::pair<BasicBlockNode *, BasicBlockNode *>;

inline void addEdge(EdgeDescriptor NewEdge) {
  revng_assert(not NewEdge.first->isCheckOrSwitch());
  NewEdge.first->addSuccessor(NewEdge.second);
  NewEdge.second->addPredecessor(NewEdge.first);
}

inline void removeEdge(EdgeDescriptor Edge) {
  revng_assert(not Edge.first->isCheckOrSwitch());
  Edge.first->removeSuccessor(Edge.second);
  Edge.second->removePredecessor(Edge.first);
}

inline void moveEdgeTarget(EdgeDescriptor Edge, BasicBlockNode *NewTarget) {
  Edge.second->removePredecessor(Edge.first);

  // Special handle for dispatcher check nodes.
  if (Edge.first->isCheckOrSwitch()) {

    // Confirm that the old target of the edge was one of the two branches.
    revng_assert((Edge.first->getTrue() == Edge.second)
                 or (Edge.first->getFalse() == Edge.second));

    // Set the appropriate successor.
    if (Edge.first->getTrue() == Edge.second) {
      Edge.first->setTrue(NewTarget);
    } else if (Edge.first->getFalse() == Edge.second) {
      Edge.first->setFalse(NewTarget);
    } else {
      revng_abort("Wrong successor for check node");
    }
  } else {
    // General case when we are not handling a dispatcher check node.
    Edge.first->removeSuccessor(Edge.second);
    Edge.first->addSuccessor(NewTarget);
    NewTarget->addPredecessor(Edge.first);
  }
}

// Helper function to find all nodes on paths between a source and a target
// node
inline std::set<BasicBlockNode *>
findReachableNodes(BasicBlockNode &Source, BasicBlockNode &Target) {

  // Add to the Targets set the original target node.
  std::set<BasicBlockNode *> Targets;
  Targets.insert(&Target);

  // Exploration stack initialization.
  std::vector<std::pair<BasicBlockNode *, size_t>> Stack;
  Stack.push_back(std::make_pair(&Source, 0));

  // Visited nodes to avoid entering in a loop.
  std::set<EdgeDescriptor> VisitedEdges;

  // Exploration.
  while (!Stack.empty()) {
    auto StackElem = Stack.back();
    Stack.pop_back();
    BasicBlockNode *Vertex = StackElem.first;
    if (StackElem.second == 0) {
      if (Targets.count(Vertex) != 0) {
        for (auto StackElem : Stack) {
          Targets.insert(StackElem.first);
        }
        continue;
      }
    }
    size_t Index = StackElem.second;
    if (Index < StackElem.first->successor_size()) {
      BasicBlockNode *NextSuccessor = Vertex->getSuccessorI(Index);
      Index++;
      Stack.push_back(std::make_pair(Vertex, Index));
      if (VisitedEdges.count(std::make_pair(Vertex, NextSuccessor)) == 0
          and NextSuccessor != &Source) {
        Stack.push_back(std::make_pair(NextSuccessor, 0));
        VisitedEdges.insert(std::make_pair(Vertex, NextSuccessor));
      }
    }
  }

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

// Debug function to serialize an AST node.
inline void dumpNode(ASTNode *Node) {
  if (auto *If = llvm::dyn_cast<IfNode>(Node)) {
    CombLogger << "\"" << If->getName() << "\" [";

    // For the label of the If node go take all the nodes in the list
    std::string ConditionalNames;
    for (BasicBlockNode *Conditional : If->conditionalNodes()) {
      ConditionalNames += Conditional->getNameStr() + ", ";
    }
    ConditionalNames.pop_back();
    ConditionalNames.pop_back();

    CombLogger << "label=\"" << ConditionalNames;
    CombLogger << "\"";
    CombLogger << ",shape=\"invhouse\",color=\"blue\"];\n";

    if (If->getThen() != nullptr) {
      CombLogger << "\"" << If->getName() << "\""
                 << " -> \"" << If->getThen()->getName() << "\""
                 << " [color=green,label=\"then\"];\n";
      dumpNode(If->getThen());
    }

    if (If->getElse() != nullptr) {
      CombLogger << "\"" << If->getName() << "\""
                 << " -> \"" << If->getElse()->getName() << "\""
                 << " [color=green,label=\"else\"];\n";
      dumpNode(If->getElse());
    }
  } else if (auto *Code = llvm::dyn_cast<CodeNode>(Node)) {

    CombLogger << "\"" << Code->getName() << "\" [";
    CombLogger << "label=\"" << Code->getName();
    CombLogger << "\"";
    CombLogger << ",shape=\"box\",color=\"red\"];\n";
  } else if (auto *Sequence = llvm::dyn_cast<SequenceNode>(Node)) {

    CombLogger << "\"" << Sequence->getName() << "\" [";
    CombLogger << "label=\"" << Sequence->getName();
    CombLogger << "\"";
    CombLogger << ",shape=\"box\",color=\"black\"];\n";

    for (ASTNode *Successor : Sequence->nodes()) {
      CombLogger << "\"" << Sequence->getName() << "\""
                 << " -> \"" << Successor->getName() << "\""
                 << " [color=green,label=\"elem\"];\n";
      dumpNode(Successor);
    }
  } else if (auto *Scs = llvm::dyn_cast<ScsNode>(Node)) {

    CombLogger << "\"" << Scs->getName() << "\" [";
    CombLogger << "label=\"" << Scs->getName();
    CombLogger << "\"";
    CombLogger << ",shape=\"circle\",color=\"black\"];\n";

    revng_assert(Scs->getBody() != nullptr);
    CombLogger << "\"" << Scs->getName() << "\""
               << " -> \"" << Scs->getBody()->getName() << "\""
               << " [color=green,label=\"body\"];\n";
    dumpNode(Scs->getBody());
  }
}

#endif // REVNGC_RESTRUCTURE_CFG_UTILS_H
