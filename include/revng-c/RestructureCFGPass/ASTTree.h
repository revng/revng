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
  using links_container = std::vector<std::unique_ptr<ASTNode>>;
  using links_iterator = typename links_container::iterator;
  using links_range = llvm::iterator_range<links_iterator>;

  using links_container_expr = std::vector<std::unique_ptr<ExprNode>>;
  using links_iterator_expr = typename links_container_expr::iterator;
  using links_range_expr = llvm::iterator_range<links_iterator_expr>;

  // TODO: consider including BasicBlockNode header file.
  using BBNodeMap = std::map<BasicBlockNode *, BasicBlockNode *>;

  using ASTNodeMap = std::map<ASTNode *, ASTNode *>;

  links_iterator begin() { return links_iterator(ASTNodeList.begin()); };
  links_iterator end() { return links_iterator(ASTNodeList.end()); };

  links_iterator_expr beginExpr() {
    return links_iterator_expr(CondExprList.begin());
  };
  links_iterator_expr endExpr() {
    return links_iterator_expr(CondExprList.end());
  };

private:
  links_container ASTNodeList;
  std::map<BasicBlockNode *, ASTNode *> NodeASTMap;
  ASTNode *RootNode;
  unsigned IDCounter = 0;
  links_container_expr CondExprList;

public:
  void addCFGNode() {}

  SequenceNode *addSequenceNode();

  unsigned getNewID() { return IDCounter++; }

  links_range nodes() { return llvm::make_range(begin(), end()); }

  links_range_expr expressions() {
    return llvm::make_range(beginExpr(), endExpr());
  }

  size_t size();

  void addASTNode(BasicBlockNode *Node, std::unique_ptr<ASTNode> &&ASTObject);

  SwitchNode *addSwitch(std::unique_ptr<ASTNode> ASTObject);

  SwitchCheckNode *addSwitchCheck(std::unique_ptr<ASTNode> ASTObject);

  ASTNode *findASTNode(BasicBlockNode *BlockNode);

  BasicBlockNode *findCFGNode(ASTNode *Node);

  void setRoot(ASTNode *Root);

  ASTNode *getRoot();

  ASTNode *copyASTNodesFrom(ASTTree &OldAST, BBNodeMap &SubstitutionMap2);

  void dumpOnFile(std::string FolderName,
                  std::string FunctionName,
                  std::string FileName);

  ExprNode *addCondExpr(std::unique_ptr<ExprNode> &&Expr);

  void copyASTNodesFrom(ASTTree &OldAST);
};

#endif // REVNGC_RESTRUCTURE_CFG_ASTTREE_H
