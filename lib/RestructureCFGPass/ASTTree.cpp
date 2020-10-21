/// \file ASTTree.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <cstdlib>

// Local libraries includes
#include "revng-c/RestructureCFGPass/ASTNode.h"
#include "revng-c/RestructureCFGPass/ASTTree.h"
#include "revng-c/RestructureCFGPass/Utils.h"

using namespace llvm;
using ASTNodeMap = std::map<ASTNode *, ASTNode *>;
using ExprNodeMap = std::map<ExprNode *, ExprNode *>;

// Helper to obtain a unique incremental counter (to give name to sequence
// nodes).
static int Counter = 1;
static std::string getID() {
  return std::to_string(Counter++);
}

SequenceNode *ASTTree::addSequenceNode() {
  ASTNodeList.emplace_back(SequenceNode::createEmpty("sequence " + getID()));

  // Set the Node ID
  ASTNodeList.back()->setID(getNewID());

  return llvm::cast<SequenceNode>(ASTNodeList.back().get());
}

size_t ASTTree::size() const {
  return ASTNodeList.size();
}

void ASTTree::addASTNode(BasicBlockNode<BasicBlock *> *Node,
                         ast_unique_ptr &&ASTObject) {
  ASTNodeList.emplace_back(std::move(ASTObject));

  ASTNode *ASTNode = ASTNodeList.back().get();

  // Set the Node ID
  ASTNode->setID(getNewID());

  bool New = BBASTMap.insert({ Node, ASTNode }).second;
  revng_assert(New);
  New = ASTBBMap.insert({ ASTNode, Node }).second;
  revng_assert(New);
}

ASTNode *ASTTree::findASTNode(BasicBlockNode<BasicBlock *> *BlockNode) {
  return BBASTMap.at(BlockNode);
}

BasicBlockNode<BasicBlock *> *ASTTree::findCFGNode(ASTNode *ASTNode) {
  auto It = ASTBBMap.find(ASTNode);
  if (It != ASTBBMap.end())
    return It->second;
  // We may return nullptr, since for example continue and break nodes do not
  // have a corresponding CFGNode.
  return nullptr;
}

void ASTTree::setRoot(ASTNode *Root) {
  RootNode = Root;
}

ASTNode *ASTTree::getRoot() const {
  return RootNode;
}

ASTNode *ASTTree::copyASTNodesFrom(ASTTree &OldAST) {
  ASTNodeMap ASTSubstitutionMap{};
  ExprNodeMap CondExprMap{};

  // Clone each ASTNode in the current AST.
  links_container::difference_type NewNodes = 0;
  for (const ast_unique_ptr &Old : OldAST.nodes()) {
    ASTNodeList.emplace_back(std::move(Old->Clone()));
    ++NewNodes;

    ASTNode *NewASTNode = ASTNodeList.back().get();
    ASTNode *OldASTNode = Old.get();

    // Set the Node ID
    NewASTNode->setID(getNewID());

    BasicBlockNode<BasicBlock *> *OldCFGNode = OldAST.findCFGNode(OldASTNode);
    if (OldCFGNode != nullptr) {
      BBASTMap.insert({ OldCFGNode, NewASTNode });
      ASTBBMap.insert({ NewASTNode, OldCFGNode });
    }
    ASTSubstitutionMap[OldASTNode] = NewASTNode;
  }

  // Clone the conditional expression nodes.
  for (const expr_unique_ptr &OldExpr : OldAST.expressions()) {
    CondExprList.emplace_back(new AtomicNode(*cast<AtomicNode>(OldExpr.get())),
                              expr_destructor());
    ExprNode *NewExpr = CondExprList.back().get();
    CondExprMap[OldExpr.get()] = NewExpr;
  }

  // Update the AST and BBNode pointers inside the newly created AST nodes,
  // to reflect the changes made. Update also the pointer to the conditional
  // expressions just cloned.
  links_iterator BeginInserted = ASTNodeList.end() - NewNodes;
  links_iterator EndInserted = ASTNodeList.end();
  using MovedIteratorRange = llvm::iterator_range<links_container::iterator>;
  MovedIteratorRange Result = llvm::make_range(BeginInserted, EndInserted);
  for (ast_unique_ptr &NewNode : Result) {
    NewNode->updateASTNodesPointers(ASTSubstitutionMap);
    if (auto *If = llvm::dyn_cast<IfNode>(NewNode.get())) {
      If->updateCondExprPtr(CondExprMap);
    }
  }

  revng_assert(ASTSubstitutionMap.count(OldAST.getRoot()) != 0);
  return ASTSubstitutionMap[OldAST.getRoot()];
}

void ASTTree::dumpOnFile(std::string FolderName,
                         std::string FunctionName,
                         std::string FileName) {

  std::ofstream ASTFile;
  std::string PathName = FolderName + "/" + FunctionName;
  mkdir(FolderName.c_str(), 0775);
  mkdir(PathName.c_str(), 0775);
  ASTFile.open(PathName + "/" + FileName + ".dot");
  if (ASTFile.is_open()) {
    ASTFile << "digraph CFGFunction {\n";
    RootNode->dump(ASTFile);
    ASTFile << "}\n";
    ASTFile.close();
  } else {
    revng_abort("Could not open file for dumping AST.");
  }
}

ExprNode *ASTTree::addCondExpr(expr_unique_ptr &&Expr) {
  CondExprList.emplace_back(std::move(Expr));
  return CondExprList.back().get();
}
