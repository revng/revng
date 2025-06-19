#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstdlib>
#include <type_traits>

#include "revng/RestructureCFG/ASTNode.h"

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
  using getPointerT = ASTNode *(*) (ast_unique_ptr &);

  static ASTNode *getPointer(ast_unique_ptr &Original) {
    return Original.get();
  }

  static_assert(std::is_same_v<decltype(&getPointer), getPointerT>);

  using links_container = std::vector<ast_unique_ptr>;
  using internal_iterator = typename links_container::iterator;
  using links_iterator = llvm::mapped_iterator<internal_iterator, getPointerT>;
  using links_range = llvm::iterator_range<links_iterator>;

  using expr_deleter_t = decltype(&ExprNode::deleteExprNode);
  using expr_destructor = std::integral_constant<expr_deleter_t,
                                                 &ExprNode::deleteExprNode>;
  using expr_unique_ptr = std::unique_ptr<ExprNode, expr_destructor>;

  using links_container_expr = std::vector<expr_unique_ptr>;
  using links_iterator_expr = typename links_container_expr::iterator;
  using links_range_expr = llvm::iterator_range<links_iterator_expr>;

  using ASTNodeMap = ASTNode::ASTNodeMap;
  using BasicBlockNodeBB = ASTNode::BasicBlockNodeBB;
  using BBNodeMap = ASTNode::BBNodeMap;

  links_iterator begin() {
    return llvm::map_iterator(ASTNodeList.begin(), getPointer);
  }
  links_iterator end() {
    return llvm::map_iterator(ASTNodeList.end(), getPointer);
  }

  links_iterator_expr beginExpr() { return CondExprList.begin(); }
  links_iterator_expr endExpr() { return CondExprList.end(); }

private:
  links_container ASTNodeList = {};
  std::map<BasicBlockNodeBB *, ASTNode *> BBASTMap = {};
  std::map<ASTNode *, BasicBlockNodeBB *> ASTBBMap = {};
  ASTNode *RootNode = nullptr;
  unsigned IDCounter = 0;
  links_container_expr CondExprList = {};

public:
  ASTTree() :
    ASTNodeList(),
    BBASTMap(),
    ASTBBMap(),
    RootNode(nullptr),
    IDCounter(0),
    CondExprList() {}

  // Default movable
  ASTTree(ASTTree &&) = default;
  ASTTree &operator=(ASTTree &&) = default;

  // Non copyable
  ASTTree(const ASTTree &) = delete;
  ASTTree &operator=(const ASTTree &) = delete;

private:
  ASTNode *addASTNodeImpl(ast_unique_ptr &&ASTObject);

public:
  SequenceNode *addSequenceNode();

  SwitchBreakNode *addSwitchBreak(SwitchNode *SN);

  unsigned getNewID() { return IDCounter++; }

  links_range nodes() { return llvm::make_range(begin(), end()); }

  links_range_expr expressions() {
    return llvm::make_range(beginExpr(), endExpr());
  }

  links_container::size_type size() const;

  void addASTNode(BasicBlockNodeBB *Node, ast_unique_ptr &&ASTObject);

  ASTNode *addASTNode(ast_unique_ptr &&ASTObject);

  void removeASTNode(ASTNode *Node);

  ASTNode *findASTNode(BasicBlockNodeBB *BlockNode);

  BasicBlockNodeBB *findCFGNode(ASTNode *Node);

  void setRoot(ASTNode *Root);

  ASTNode *getRoot() const;

  ASTNode *copyASTNodesFrom(ASTTree &OldAST);

  /// Dump a GraphViz file on a file using an absolute path
  debug_function void dumpASTOnFile(const std::string &FileName) const;

  /// Dump a GraphViz file on a file using an absolute path
  debug_function void dumpASTOnFile(const char *FName) const {
    return dumpASTOnFile(std::string(FName));
  }

  /// Dump a GraphViz file on a file representing this function
  debug_function void dumpASTOnFile(const std::string &FunctionName,
                                    const std::string &FolderName,
                                    const std::string &FileName) const;

  ExprNode *addCondExpr(expr_unique_ptr &&Expr);
};
