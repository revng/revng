/// \file MetaRegion.h

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#ifndef REVNGC_RESTRUCTURE_CFG_META_REGIONS_H
#define REVNGC_RESTRUCTURE_CFG_META_REGIONS_H

// std includes
#include <utility>
#include <set>
#include <memory>
#include <vector>

// LLVM includes
#include <llvm/ADT/iterator_range.h>

// local libraries include
#include "revng-c/RestructureCFGPass/RegionCFGTree.h"

class BasicBlockNode;

/// \brief The MetaRegion class, a wrapper for a set of nodes.
class MetaRegion {

public:
  using links_container = std::set<BasicBlockNode *>;
  using links_iterator = typename links_container::iterator;
  using links_const_iterator = typename links_container::const_iterator;
  using links_range = llvm::iterator_range<links_iterator>;
  using links_const_range = llvm::iterator_range<links_const_iterator>;
  using EdgeDescriptor = std::pair<BasicBlockNode *, BasicBlockNode *>;

  inline links_iterator begin() { return Nodes.begin(); };
  inline links_const_iterator cbegin() const { return Nodes.cbegin(); };
  inline links_iterator end() { return Nodes.end(); };
  inline links_const_iterator cend() const { return Nodes.cend(); };

private:
  int Index;
  links_container Nodes;
  MetaRegion *ParentRegion;
  bool IsSCS;
  CFG Graph;

public:
  MetaRegion(int Index, std::set<BasicBlockNode *> &Nodes, bool IsSCS = false) :
    Index(Index), Nodes(Nodes), IsSCS(IsSCS) {}

  int getIndex() const { return Index; }

  void replaceNodes(std::vector<std::unique_ptr<BasicBlockNode>> &NewNodes);

  void updateNodes(std::set<BasicBlockNode *> &Removal,
                   BasicBlockNode *Collapsed,
                   std::vector<BasicBlockNode *> Dispatcher);

  void setParent(MetaRegion *Parent) { ParentRegion = Parent; }

  MetaRegion *getParent() { return ParentRegion; }

  std::set<BasicBlockNode *> &getNodes() { return Nodes; }

  size_t nodes_size() const { return Nodes.size(); }

  links_const_range nodes() const {
    return llvm::make_range(Nodes.begin(), Nodes.end());
  }

  links_range nodes() {
    return llvm::make_range(Nodes.begin(), Nodes.end());
  }

  std::set<BasicBlockNode *> getSuccessors();

  std::set<EdgeDescriptor> getOutEdges();

  std::set<EdgeDescriptor> getInEdges();

  bool intersectsWith(MetaRegion &Other) const;

  bool isSubSet(MetaRegion &Other) const;

  bool isSuperSet(MetaRegion &Other) const;

  bool nodesEquality(MetaRegion &Other) const;

  void mergeWith(MetaRegion &Other) {
    std::set<BasicBlockNode *> &OtherNodes = Other.getNodes();
    Nodes.insert(OtherNodes.begin(), OtherNodes.end());
  }

  bool isSCS() const { return IsSCS; }

  bool containsNode(BasicBlockNode *Node) const { return Nodes.count(Node); }

  void insertNode(BasicBlockNode *NewNode) { Nodes.insert(NewNode); }

  void removeNode(BasicBlockNode *Node) { Nodes.erase(Node); }

  CFG &getGraph() { return Graph; }
};

#endif // REVNGC_RESTRUCTURE_CFG_META_REGIONS_H
