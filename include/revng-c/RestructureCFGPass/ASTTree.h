#ifndef REVNGC_RESTRUCTURE_CFG_ASTTREE_H
#define REVNGC_RESTRUCTURE_CFG_ASTTREE_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <cstdlib>

// Local includes
#include "ASTNode.h"

class ASTNode;
class BasicBlockNode;
class SequenceNode;

class ASTTree {

public:
  using links_container  = std::vector<std::unique_ptr<ASTNode>>;
  using links_iterator = typename links_container::iterator;
  using links_range = llvm::iterator_range<links_iterator>;

  links_iterator begin() { return links_iterator(ASTNodeList.begin()); };
  links_iterator end() { return links_iterator(ASTNodeList.end()); };

private:
  std::map<BasicBlockNode *, std::unique_ptr<ASTNode>> NodeASTMap;
  links_container ASTNodeList;

public:

  void addCFGNode() {}

  SequenceNode *addSequenceNode();

  links_range nodes() {
    return llvm::make_range(begin(), end());
  }

  size_t size();

  void addASTNode(BasicBlockNode *Node, std::unique_ptr<ASTNode>&& ASTObject);

  ASTNode *findASTNode(BasicBlockNode *BlockNode);

};

#endif // REVNGC_RESTRUCTURE_CFG_ASTTREE_H
