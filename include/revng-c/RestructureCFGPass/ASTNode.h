#ifndef REVNGC_RESTRUCTURE_CFG_ASTNODE_H
#define REVNGC_RESTRUCTURE_CFG_ASTNODE_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <cstdlib>

// LLVM includes
#include <llvm/Support/Casting.h>

// local includes
#include "revng-c/RestructureCFGPass/BasicBlockNode.h"

// forward declarations
namespace llvm {
class BasicBlock;
}

class BasicBlockNode;

using BBNodeMap = std::map<BasicBlockNode *, BasicBlockNode *>;

class ASTNode {

public:
  enum NodeKind { NK_Code, NK_Break, NK_Continue, NK_If, NK_Scs, NK_List };

  using ASTNodeMap = std::map<ASTNode *, ASTNode *>;

private:
  const NodeKind Kind;

protected:
  BasicBlockNode *CFGNode;
  bool Processed = false;
  std::string Name;
  ASTNode *Successor = nullptr;

  /// Unique Node ID inside a ASTNode, useful for printing to graphviz
  /// This field is initialized to 0, and will be re-assigned when the ASTNode
  /// will be inserted in an ASTTree.
  unsigned ID = 0;

public:
  ASTNode(NodeKind K, const std::string &Name, ASTNode *Successor = nullptr) :
    Kind(K),
    CFGNode(nullptr),
    Name(Name),
    Successor(Successor) {}

  ASTNode(NodeKind K, BasicBlockNode *CFGNode, ASTNode *Successor = nullptr) :
    Kind(K),
    CFGNode(CFGNode),
    Name(CFGNode->getNameStr()),
    Successor(Successor) {}

  virtual ~ASTNode() {}

  virtual ASTNode *Clone() = 0;

public:
  NodeKind getKind() const { return Kind; }

  std::string getName() {
    return "ID:" + std::to_string(getID()) + " Name:" + Name;
  }

  void setID(unsigned NewID) { ID = NewID; }

  unsigned getID() const { return ID; }

  virtual void dump(std::ofstream &ASTFile) = 0;

  BasicBlockNode *getCFGNode() { return CFGNode; }

  ASTNode *getSuccessor() { return Successor; }

  bool isEmpty() {

    // Check if the corresponding CFGNode is a dummy node. In case we do not
    // have a corresponding CFGNode (e.g., a sequence node), assume that this
    // property is not verified
    if (CFGNode != nullptr) {
      return CFGNode->isEmpty();
    } else {
      return false;
    }
  }

  llvm::BasicBlock *getOriginalBB() {
    if (CFGNode != nullptr) {
      return CFGNode->getBasicBlock();
    } else {
      return nullptr;
    }
  }

  virtual bool isEqual(ASTNode *Node) = 0;

  virtual BasicBlockNode *getFirstCFG() = 0;

  virtual void updateBBNodePointers(BBNodeMap &SubstitutionMap) = 0;

  virtual void updateASTNodesPointers(ASTNodeMap &SubstitutionMap) = 0;
};

class CodeNode : public ASTNode {

public:
  CodeNode(BasicBlockNode *CFGNode, ASTNode *Successor) :
    ASTNode(NK_Code, CFGNode, Successor) {}

public:
  static bool classof(const ASTNode *N) { return N->getKind() == NK_Code; }

  bool isEqual(ASTNode *Node);

  void dump(std::ofstream &ASTFile);

  BasicBlockNode *getFirstCFG();

  void updateBBNodePointers(BBNodeMap &SubstitutionMap);

  void updateASTNodesPointers(ASTNodeMap &SubstitutionMap);

  ASTNode *Clone() { return new CodeNode(*this); }
};

class IfNode : public ASTNode {

public:
  using links_container = std::vector<BasicBlockNode *>;
  using links_iterator = typename links_container::iterator;
  using links_range = llvm::iterator_range<links_iterator>;

private:
  ASTNode *Then;
  ASTNode *Else;
  std::vector<BasicBlockNode *> ConditionalNodes;
  bool NegatedCondition = false;

public:
  IfNode(BasicBlockNode *CFGNode,
         ASTNode *Then,
         ASTNode *Else,
         ASTNode *PostDom) :
    ASTNode(NK_If, CFGNode, PostDom),
    Then(Then),
    Else(Else) {
    ConditionalNodes.push_back(CFGNode);
  }

public:
  static bool classof(const ASTNode *N) { return N->getKind() == NK_If; }

  llvm::BasicBlock *getUniqueCondBlock() {
    revng_assert(ConditionalNodes.size() == 1);
    BasicBlockNode *N = ConditionalNodes[0];
    revng_assert(N->isCode() or N->isCheck());
    return N->getBasicBlock();
  }

  ASTNode *getThen() { return Then; }

  ASTNode *getElse() { return Else; }

  void setThen(ASTNode *Node) { Then = Node; }

  void setElse(ASTNode *Node) { Else = Node; }

  bool hasThen() {
    if (Then != nullptr) {
      return true;
    }
    return false;
  }

  bool hasElse() {
    if (Else != nullptr) {
      return true;
    }
    return false;
  }

  bool hasBothBranches() {
    if ((Then != nullptr) and (Else != nullptr)) {
      return true;
    } else {
      return false;
    }
  }

  links_range conditionalNodes() {
    return llvm::make_range(ConditionalNodes.begin(), ConditionalNodes.end());
  }

  void addConditionalNodesFrom(IfNode *Other);

  bool isEqual(ASTNode *Node);

  void dump(std::ofstream &ASTFile);

  BasicBlockNode *getFirstCFG();

  void updateBBNodePointers(BBNodeMap &SubstitutionMap);

  void updateASTNodesPointers(ASTNodeMap &SubstitutionMap);

  ASTNode *Clone() { return new IfNode(*this); }

  void negateCondition() { NegatedCondition = true; }

  bool conditionNegated() const { return NegatedCondition; }
};

class ScsNode : public ASTNode {

public:
  enum class Type {
    Standard,
    While,
    DoWhile,
  };

private:
  ASTNode *Body;
  Type LoopType = Type::Standard;
  IfNode *RelatedCondition = nullptr;

public:
  ScsNode(BasicBlockNode *CFGNode, ASTNode *Body) :
    ASTNode(NK_Scs, CFGNode),
    Body(Body) {}
  ScsNode(BasicBlockNode *CFGNode, ASTNode *Body, ASTNode *Successor) :
    ASTNode(NK_Scs, CFGNode, Successor),
    Body(Body) {}

public:
  static bool classof(const ASTNode *N) { return N->getKind() == NK_Scs; }

  ASTNode *getBody() { return Body; }

  void setBody(ASTNode *Node) { Body = Node; }

  bool isEqual(ASTNode *Node);

  void dump(std::ofstream &ASTFile);

  BasicBlockNode *getFirstCFG();

  void updateBBNodePointers(BBNodeMap &SubstitutionMap);

  void updateASTNodesPointers(ASTNodeMap &SubstitutionMap);

  ASTNode *Clone() { return new ScsNode(*this); }

  bool isStandard() { return LoopType == Type::Standard; }

  bool isWhile() { return LoopType == Type::While; }

  bool isDoWhile() { return LoopType == Type::DoWhile; }

  void setWhile(IfNode *Condition) {
    LoopType = Type::While;
    RelatedCondition = Condition;
  }

  void setDoWhile(IfNode *Condition) {
    LoopType = Type::DoWhile;
    RelatedCondition = Condition;
  }

  IfNode *getRelatedCondition () {
    revng_assert(LoopType == Type::While or LoopType == Type::DoWhile);
    revng_assert(RelatedCondition != nullptr);

    return RelatedCondition;
  }

};

class SequenceNode : public ASTNode {

public:
  using links_container = std::vector<ASTNode *>;
  using links_iterator = typename links_container::iterator;
  using links_range = llvm::iterator_range<links_iterator>;

private:
  links_container NodeList;

public:
  SequenceNode(std::string Name) : ASTNode(NK_List, Name) {}
  SequenceNode(BasicBlockNode *CFGNode) : ASTNode(NK_List, CFGNode) {}

public:
  static bool classof(const ASTNode *N) { return N->getKind() == NK_List; }

  links_range nodes() {
    return llvm::make_range(NodeList.begin(), NodeList.end());
  }

  void addNode(ASTNode *Node) {
    NodeList.push_back(Node);
    if (Node->getSuccessor() != nullptr) {
      this->addNode(Node->getSuccessor());
    }
  }

  void removeNode(ASTNode *Node) {
    NodeList.erase(std::remove(NodeList.begin(), NodeList.end(), Node),
                   NodeList.end());
  }

  int listSize() { return NodeList.size(); }

  ASTNode *getNodeN(int N) { return NodeList[N]; }

  bool isEqual(ASTNode *Node);

  void dump(std::ofstream &ASTFile);

  BasicBlockNode *getFirstCFG();

  void updateBBNodePointers(BBNodeMap &SubstitutionMap);

  void updateASTNodesPointers(ASTNodeMap &SubstitutionMap);

  ASTNode *Clone() { return new SequenceNode(*this); }
};


class ContinueNode : public ASTNode {
private:
  IfNode *ComputationIf = nullptr;

public:
  ContinueNode() : ASTNode(NK_Continue, "continue"){};

  static bool classof(const ASTNode *N) { return N->getKind() == NK_Continue; }

  ASTNode *Clone() { return new ContinueNode(*this); }

  void dump(std::ofstream &ASTFile);

  bool isEqual(ASTNode *Node) { return llvm::isa<ContinueNode>(Node); }

  BasicBlockNode *getFirstCFG() { return nullptr; };

  void updateBBNodePointers(BBNodeMap &SubstitutionMap) {}

  void updateASTNodesPointers(ASTNodeMap &SubstitutionMap) {}

  bool hasComputation() { return ComputationIf != nullptr; };

  void addComputationIfNode(IfNode *ComputationIfNode);

  IfNode *getComputationIfNode();
};

class BreakNode : public ASTNode {

public:
  BreakNode() : ASTNode(NK_Break, "break"){};

  static bool classof(const ASTNode *N) { return N->getKind() == NK_Break; }

  ASTNode *Clone() { return new BreakNode(*this); }

  void dump(std::ofstream &ASTFile);

  bool isEqual(ASTNode *Node) { return llvm::isa<BreakNode>(Node); }

  BasicBlockNode *getFirstCFG() { return nullptr; };

  void updateBBNodePointers(BBNodeMap &SubstitutionMap) {}

  void updateASTNodesPointers(ASTNodeMap &SubstitutionMap) {}
};

#endif // define REVNGC_RESTRUCTURE_CFG_ASTNODE_H
