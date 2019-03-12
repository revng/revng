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

using ASTNodeMap = std::map<ASTNode *, ASTNode *>;

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

size_t ASTTree::size() {
  return ASTNodeList.size();
}

void ASTTree::addASTNode(BasicBlockNode *Node,
                         std::unique_ptr<ASTNode> &&ASTObject) {
  ASTNodeList.emplace_back(std::move(ASTObject));

  ASTNode *ASTNode = ASTNodeList.back().get();

  // Set the Node ID
  ASTNode->setID(getNewID());

  auto InsertResult = NodeASTMap.insert(std::make_pair(Node, ASTNode));
}

SwitchNode *ASTTree::addSwitch(std::unique_ptr<ASTNode> &&ASTObject) {
  ASTNodeList.emplace_back(std::move(ASTObject));

  // Set the Node ID
  ASTNodeList.back()->setID(getNewID());

  return llvm::cast<SwitchNode>(ASTNodeList.back().get());
}

ASTNode *ASTTree::findASTNode(BasicBlockNode *BlockNode) {
  revng_assert(NodeASTMap.count(BlockNode) != 0);
  ASTNode *ASTPointer = NodeASTMap[BlockNode];
  return ASTPointer;
}

void ASTTree::setRoot(ASTNode *Root) {
  RootNode = Root;
}

ASTNode *ASTTree::getRoot() {
  return RootNode;
}

ASTNode *
ASTTree::copyASTNodesFrom(ASTTree &OldAST, BBNodeMap &SubstitutionMap) {
  size_t NumCurrNodes = size();
  ASTNodeMap ASTSubstitutionMap{};

  // Clone each ASTNode in the current AST.
  for (std::unique_ptr<ASTNode> &Old : OldAST.nodes()) {
    // ASTNodeList.emplace_back(std::make_unique<ASTNode>(*Old));
    ASTNodeList.emplace_back(std::move(Old->Clone()));

    // Set the Node ID
    ASTNodeList.back()->setID(getNewID());

    BasicBlockNode *OldCFGNode = Old->getCFGNode();
    if (OldCFGNode != nullptr) {
      NodeASTMap.insert(std::make_pair(OldCFGNode, ASTNodeList.back().get()));
    }
    ASTNode *New = ASTNodeList.back().get();
    ASTSubstitutionMap[Old.get()] = New;
  }

  // Update the AST and BBNode pointers inside the newly created AST nodes,
  // to reflect the changes made.
  links_container::iterator BeginInserted = ASTNodeList.begin() + NumCurrNodes;
  links_container::iterator EndInserted = ASTNodeList.end();
  using MovedIteratorRange = llvm::iterator_range<links_container::iterator>;
  MovedIteratorRange Result = llvm::make_range(BeginInserted, EndInserted);
  for (std::unique_ptr<ASTNode> &NewNode : Result) {
    NewNode->updateASTNodesPointers(ASTSubstitutionMap);
    NewNode->updateBBNodePointers(SubstitutionMap);
  }

  // Update the map between `BasicBlockNode` and `ASTNode`.
  for (auto SubIt : SubstitutionMap) {
    BasicBlockNode *OldBB = SubIt.first;
    BasicBlockNode *NewBB = SubIt.second;
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
  ASTFile << "digraph CFGFunction {\n";
  RootNode->dump(ASTFile);
  ASTFile << "}\n";
  ASTFile.close();
}

// Helper function that visit an AST tree and creates the sequence nodes
ASTNode *createSequence(ASTTree &Tree, ASTNode *RootNode) {
  SequenceNode *RootSequenceNode = Tree.addSequenceNode();
  RootSequenceNode->addNode(RootNode);

  for (ASTNode *Node : RootSequenceNode->nodes()) {
    if (auto *If = llvm::dyn_cast<IfNode>(Node)) {
      If->setThen(createSequence(Tree, If->getThen()));
      If->setElse(createSequence(Tree, If->getElse()));
    } else if (auto *Code = llvm::dyn_cast<CodeNode>(Node)) {
      // TODO: confirm that doesn't make sense to process a code node.
    } else if (auto *Scs = llvm::dyn_cast<ScsNode>(Node)) {
      // TODO: confirm that this phase is not needed since the processing is
      //       done inside the processing of each SCS region.
    }
  }

  return RootSequenceNode;
}

// Helper function which simplifies sequence nodes composed by a single AST
// node.
ASTNode *simplifyAtomicSequence(ASTNode *RootNode) {

  if (auto *Sequence = llvm::dyn_cast<SequenceNode>(RootNode)) {
    if (Sequence->listSize() == 0) {
      RootNode = nullptr;
    } else if (Sequence->listSize() == 1) {
      RootNode = Sequence->getNodeN(0);
      RootNode = simplifyAtomicSequence(RootNode);
    } else {
      for (ASTNode *&Node : Sequence->nodes()) {
        Node = simplifyAtomicSequence(Node);
      }
    }
  } else if (auto *If = llvm::dyn_cast<IfNode>(RootNode)) {
    If->setThen(simplifyAtomicSequence(If->getThen()));
    If->setElse(simplifyAtomicSequence(If->getElse()));
  } else if (auto *Scs = llvm::dyn_cast<ScsNode>(RootNode)) {
    // TODO: check if this is not needed as the simplification is done for
    //       each SCS region.
  }

  return RootNode;
}
