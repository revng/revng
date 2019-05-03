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
  ASTNodeList.emplace_back(new SequenceNode("sequence " + getID()));

  // Set the Node ID
  ASTNodeList.back()->setID(getNewID());

  return llvm::cast<SequenceNode>(ASTNodeList.back().get());
}

size_t ASTTree::size() const {
  return ASTNodeList.size();
}

void ASTTree::addASTNode(BasicBlockNode<BasicBlock *> *Node,
                         std::unique_ptr<ASTNode> &&ASTObject) {
  ASTNodeList.emplace_back(std::move(ASTObject));

  ASTNode *ASTNode = ASTNodeList.back().get();

  // Set the Node ID
  ASTNode->setID(getNewID());

  auto InsertResult = NodeASTMap.insert(std::make_pair(Node, ASTNode));
}

SwitchCheckNode *ASTTree::addSwitchCheck(std::unique_ptr<ASTNode> ASTObject) {
  ASTNodeList.emplace_back(std::move(ASTObject));

  // Set the Node ID
  ASTNodeList.back()->setID(getNewID());

  return llvm::cast<SwitchCheckNode>(ASTNodeList.back().get());
}

SwitchNode *ASTTree::addSwitch(std::unique_ptr<ASTNode> ASTObject) {
  ASTNodeList.emplace_back(std::move(ASTObject));

  // Set the Node ID
  ASTNodeList.back()->setID(getNewID());

  return llvm::cast<SwitchNode>(ASTNodeList.back().get());
}

ASTNode *ASTTree::findASTNode(BasicBlockNode<BasicBlock *> *BlockNode) {
  revng_assert(NodeASTMap.count(BlockNode) != 0);
  ASTNode *ASTPointer = NodeASTMap[BlockNode];
  return ASTPointer;
}

// TODO: This is higly inefficient (it is a linear scan over a std::map).
//       Consider keeping another map for the inverse direction (and updating
//       it).
BasicBlockNode<BasicBlock *> *ASTTree::findCFGNode(ASTNode *Node) {
  BasicBlockNode<BasicBlock *> *Result = nullptr;
  for (auto MapIt = NodeASTMap.begin(); MapIt != NodeASTMap.end(); MapIt++) {
    if (MapIt->second == Node) {
      Result = MapIt->first;
    }
  }

  // We may return nullptr, since for example continue and break nodes do not
  // have a corresponding CFGNode.
  return Result;
}

void ASTTree::setRoot(ASTNode *Root) {
  RootNode = Root;
}

ASTNode *ASTTree::getRoot() const {
  return RootNode;
}

ASTNode *
ASTTree::copyASTNodesFrom(ASTTree &OldAST, BBNodeMap &SubstitutionMap) {
  size_t NumCurrNodes = size();
  ASTNodeMap ASTSubstitutionMap{};
  ExprNodeMap CondExprMap{};

  // Clone each ASTNode in the current AST.
  for (std::unique_ptr<ASTNode> &Old : OldAST.nodes()) {
    ASTNodeList.emplace_back(std::move(Old->Clone()));

    // Set the Node ID
    ASTNodeList.back()->setID(getNewID());

    BasicBlockNode<BasicBlock *> *OldCFGNode = OldAST.findCFGNode(Old.get());
    if (OldCFGNode != nullptr) {
      NodeASTMap.insert(std::make_pair(OldCFGNode, ASTNodeList.back().get()));
    }
    ASTNode *New = ASTNodeList.back().get();
    ASTSubstitutionMap[Old.get()] = New;
  }

  // Clone the conditional expression nodes.
  for (std::unique_ptr<ExprNode> &OldExpr : OldAST.expressions()) {
    CondExprList.emplace_back(new AtomicNode(*cast<AtomicNode>(OldExpr.get())));
    ExprNode *NewExpr = CondExprList.back().get();
    CondExprMap[OldExpr.get()] = NewExpr;
  }

  // Update the AST and BBNode pointers inside the newly created AST nodes,
  // to reflect the changes made. Update also the pointer to the conditional
  // expressions just cloned.
  links_container::iterator BeginInserted = ASTNodeList.begin() + NumCurrNodes;
  links_container::iterator EndInserted = ASTNodeList.end();
  using MovedIteratorRange = llvm::iterator_range<links_container::iterator>;
  MovedIteratorRange Result = llvm::make_range(BeginInserted, EndInserted);
  for (std::unique_ptr<ASTNode> &NewNode : Result) {
    NewNode->updateASTNodesPointers(ASTSubstitutionMap);
    if (auto *If = llvm::dyn_cast<IfNode>(NewNode.get())) {
      If->updateCondExprPtr(CondExprMap);
    }
  }

  // Update the map between `BasicBlockNode` and `ASTNode`.
  for (auto SubIt : SubstitutionMap) {
    BasicBlockNode<BasicBlock *> *OldBB = SubIt.first;
    BasicBlockNode<BasicBlock *> *NewBB = SubIt.second;
    auto MapIt = NodeASTMap.find(OldBB);

    // HACK:: update the key of the NodeASTMap
    bool isBreakOrContinue = OldBB->isContinue() or OldBB->isBreak();
    if (not isBreakOrContinue) {
      revng_assert(MapIt != NodeASTMap.end());
      std::swap(NodeASTMap[NewBB], MapIt->second);
      NodeASTMap.erase(MapIt);
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

ExprNode *ASTTree::addCondExpr(std::unique_ptr<ExprNode> &&Expr) {
  CondExprList.emplace_back(std::move(Expr));
  return CondExprList.back().get();
}
