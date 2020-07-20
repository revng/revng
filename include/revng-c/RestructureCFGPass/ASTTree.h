#ifndef REVNGC_RESTRUCTURE_CFG_ASTTREE_H
#define REVNGC_RESTRUCTURE_CFG_ASTTREE_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <cstdlib>
#include <type_traits>

// local libraries includes
#include "revng-c/RestructureCFGPass/ASTNode.h"

// Forward declarations.
class ASTNode;

template<class NodeT>
class BasicBlockNode;

class SequenceNode;

class ASTTree {

public:
  using ast_deleter_t = decltype(&ASTNode::deleteASTNode);
  using ast_destructor = std::integral_constant<ast_deleter_t,
                                                &ASTNode::deleteASTNode>;
  using ast_unique_ptr = std::unique_ptr<ASTNode, ast_destructor>;

  using links_container = std::vector<ast_unique_ptr>;
  using links_iterator = typename links_container::iterator;
  using links_range = llvm::iterator_range<links_iterator>;

  using expr_deleter_t = decltype(&ExprNode::deleteExprNode);
  using expr_destructor = std::integral_constant<expr_deleter_t,
                                                 &ExprNode::deleteExprNode>;
  using expr_unique_ptr = std::unique_ptr<ExprNode, expr_destructor>;

  using links_container_expr = std::vector<expr_unique_ptr>;
  using links_iterator_expr = typename links_container_expr::iterator;
  using links_range_expr = llvm::iterator_range<links_iterator_expr>;

  // TODO: consider including BasicBlockNode header file.
  using ASTNodeMap = ASTNode::ASTNodeMap;
  using BasicBlockNodeBB = ASTNode::BasicBlockNodeBB;
  using BBNodeMap = ASTNode::BBNodeMap;

  links_iterator begin() { return links_iterator(ASTNodeList.begin()); }
  links_iterator end() { return links_iterator(ASTNodeList.end()); }

  links_iterator_expr beginExpr() {
    return links_iterator_expr(CondExprList.begin());
  }
  links_iterator_expr endExpr() {
    return links_iterator_expr(CondExprList.end());
  }

private:
  links_container ASTNodeList;
  std::map<BasicBlockNodeBB *, ASTNode *> NodeASTMap;
  ASTNode *RootNode;
  unsigned IDCounter = 0;
  links_container_expr CondExprList;

public:
  SequenceNode *addSequenceNode();

  SwitchBreakNode *addSwitchBreak() {
    ASTNodeList.emplace_back(new SwitchBreakNode());
    ASTNodeList.back()->setID(getNewID());
    return llvm::cast<SwitchBreakNode>(ASTNodeList.back().get());
  }

  unsigned getNewID() { return IDCounter++; }

  links_range nodes() { return llvm::make_range(begin(), end()); }

  links_range_expr expressions() {
    return llvm::make_range(beginExpr(), endExpr());
  }

  links_container::size_type size() const;

  void addASTNode(BasicBlockNodeBB *Node, ast_unique_ptr &&ASTObject);

  ASTNode *findASTNode(BasicBlockNodeBB *BlockNode);

  BasicBlockNodeBB *findCFGNode(ASTNode *Node);

  void setRoot(ASTNode *Root);

  ASTNode *getRoot() const;

  ASTNode *copyASTNodesFrom(ASTTree &OldAST, BBNodeMap &SubstitutionMap2);

  void dumpOnFile(std::string FolderName,
                  std::string FunctionName,
                  std::string FileName);

  ExprNode *addCondExpr(expr_unique_ptr &&Expr);

  void copyASTNodesFrom(ASTTree &OldAST);
};

#endif // REVNGC_RESTRUCTURE_CFG_ASTTREE_H
