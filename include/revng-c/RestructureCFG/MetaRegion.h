#pragma once

//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include <memory>
#include <set>
#include <utility>
#include <vector>

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/iterator_range.h"

#include "revng-c/RestructureCFG/RegionCFGTreeBB.h"

template<class NodeT>
class BasicBlockNode;

/// The MetaRegion class, a wrapper for a set of nodes.
template<class NodeT>
class MetaRegion {

public:
  using BasicBlockNodeT = typename BasicBlockNode<NodeT>::BasicBlockNodeT;
  using BasicBlockNodeTSet = std::set<BasicBlockNodeT *>;
  using BasicBlockNodeTVect = std::vector<BasicBlockNodeT *>;
  using BasicBlockNodeTUPVect = std::vector<std::unique_ptr<BasicBlockNodeT>>;
  using EdgeDescriptor = typename BasicBlockNode<NodeT>::EdgeDescriptor;

  using links_container = std::set<BasicBlockNodeT *>;
  using links_iterator = typename links_container::iterator;
  using links_const_iterator = typename links_container::const_iterator;
  using links_range = llvm::iterator_range<links_iterator>;
  using links_const_range = llvm::iterator_range<links_const_iterator>;

  inline links_iterator begin() { return Nodes.begin(); }
  inline links_const_iterator cbegin() const { return Nodes.cbegin(); }
  inline links_iterator end() { return Nodes.end(); }
  inline links_const_iterator cend() const { return Nodes.cend(); }

  using BasicBlockNodeRPOT = llvm::ReversePostOrderTraversal<BasicBlockNodeT *>;

private:
  int Index;
  links_container Nodes;
  MetaRegion<NodeT> *ParentRegion;
  bool IsSCS;

public:
  MetaRegion(int Index, BasicBlockNodeTSet &Nodes, bool IsSCS = false) :
    Index(Index), Nodes(Nodes), IsSCS(IsSCS) {}

  int getIndex() const { return Index; }

  void replaceNodes(BasicBlockNodeTUPVect &NewNodes);

  void updateNodes(const BasicBlockNodeTSet &Removal,
                   BasicBlockNodeT *Collapsed,
                   BasicBlockNodeT *ExitDispatcher,
                   const BasicBlockNodeTVect &DefaultEntrySet,
                   const BasicBlockNodeTVect &OutlinedNodes);

  void setParent(MetaRegion<NodeT> *Parent) { ParentRegion = Parent; }

  MetaRegion *getParent() const { return ParentRegion; }

  std::set<BasicBlockNode<NodeT> *> &getNodes() { return Nodes; }

  const std::set<BasicBlockNode<NodeT> *> &getNodes() const { return Nodes; }

  size_t nodes_size() const { return Nodes.size(); }

  links_const_range nodes() const {
    return llvm::make_range(Nodes.begin(), Nodes.end());
  }

  links_range nodes() { return llvm::make_range(Nodes.begin(), Nodes.end()); }

  std::set<BasicBlockNode<NodeT> *> getSuccessors();

  std::set<EdgeDescriptor> getOutEdges();

  std::set<EdgeDescriptor> getInEdges();

  bool intersectsWith(MetaRegion<NodeT> &Other) const;

  bool isSubSet(MetaRegion<NodeT> &Other) const;

  bool isSuperSet(MetaRegion<NodeT> &Other) const;

  bool nodesEquality(MetaRegion<NodeT> &Other) const;

  void mergeWith(MetaRegion<NodeT> &Other) {
    BasicBlockNodeTSet &OtherNodes = Other.getNodes();
    Nodes.insert(OtherNodes.begin(), OtherNodes.end());
  }

  bool isSCS() const { return IsSCS; }

  bool containsNode(BasicBlockNodeT *Node) const {
    return Nodes.contains(Node);
  }

  void insertNode(BasicBlockNodeT *NewNode) { Nodes.insert(NewNode); }

  void removeNode(BasicBlockNodeT *Node) { Nodes.erase(Node); }
};
