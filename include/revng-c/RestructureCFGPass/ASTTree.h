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

  // TODO: consider including BasicBlockNode header file.
  using BBNodeMap = std::map<BasicBlockNode *, BasicBlockNode *>;

  using ASTNodeMap = std::map<ASTNode *, ASTNode *>;

  links_iterator begin() { return links_iterator(ASTNodeList.begin()); };
  links_iterator end() { return links_iterator(ASTNodeList.end()); };

private:
  links_container ASTNodeList;
  std::map<BasicBlockNode *, ASTNode *> NodeASTMap;
  ASTNode *RootNode;
  unsigned IDCounter = 0;

public:

  void addCFGNode() {}

  SequenceNode *addSequenceNode();

  unsigned getNewID() { return IDCounter++; }

  links_range nodes() {
    return llvm::make_range(begin(), end());
  }

  size_t size();

  void addASTNode(BasicBlockNode *Node, std::unique_ptr<ASTNode>&& ASTObject);

  ASTNode *findASTNode(BasicBlockNode *BlockNode);

  void setRoot(ASTNode *Root);

  ASTNode *getRoot();

  ASTNode *copyASTNodesFrom(ASTTree &OldAST, BBNodeMap &SubstitutionMap2);

  void dumpOnFile(std::string FolderName,
                  std::string FunctionName,
                  std::string FileName);

};

#endif // REVNGC_RESTRUCTURE_CFG_ASTTREE_H
