#ifndef REVNGC_RESTRUCTURE_CFG_ASTNODE_H
#define REVNGC_RESTRUCTURE_CFG_ASTNODE_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <cstdlib>

// local includes
#include "revng-c/RestructureCFGPass/BasicBlockNode.h"

// forward declarations
namespace llvm {
class BasicBlock;
}

class BasicBlockNode;

class ASTNode {

public:
  enum NodeKind {
    NK_Code,
    NK_If,
    NK_Scs,
    NK_List
  };

private:
  const NodeKind Kind;

protected:
  BasicBlockNode *CFGNode;
  bool Processed = false;
  std::string Name;
  ASTNode *Successor = nullptr;

public:
  ASTNode(NodeKind K, BasicBlockNode *CFGNode) :
    Kind(K), CFGNode(CFGNode), Name(CFGNode->getNameStr()) {}

  ASTNode(NodeKind K, std::string Name) :
    Kind(K),
    CFGNode(nullptr),
    Name(Name) {}

  ASTNode(NodeKind K, BasicBlockNode *CFGNode, ASTNode *Successor) :
    Kind(K),
    CFGNode(CFGNode),
    Name(CFGNode->getNameStr()),
    Successor(Successor) {}

  virtual ~ASTNode() {}

public:

  NodeKind getKind() const { return Kind; }

  std::string getName() {
    return Name;
  }

  virtual void dump(std::ofstream &ASTFile) = 0;

  BasicBlockNode *getCFGNode() {
    return CFGNode;
  }

  ASTNode *getSuccessor() {
    return Successor;
  }

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

private:

};

class CodeNode : public ASTNode {

public:
  CodeNode(BasicBlockNode *CFGNode, ASTNode *Successor) :
    ASTNode(NK_Code, CFGNode, Successor) {}

public:
  static bool classof(const ASTNode *N) {
    return N->getKind() == NK_Code;
  }

  bool isEqual(ASTNode *Node);

  void dump(std::ofstream &ASTFile);

  BasicBlockNode *getFirstCFG();

};

class IfNode : public ASTNode {

public:
  using links_container  = std::vector<BasicBlockNode *>;
  using links_iterator = typename links_container::iterator;
  using links_range = llvm::iterator_range<links_iterator>;

private:
  ASTNode *Then;
  ASTNode *Else;
  std::vector<BasicBlockNode *> ConditionalNodes;

public:
  IfNode(BasicBlockNode *CFGNode,
         ASTNode *Then,
         ASTNode *Else,
         ASTNode *PostDom) :
    ASTNode(NK_If, CFGNode, PostDom), Then(Then), Else(Else) {
      ConditionalNodes.push_back(CFGNode);
    }

public:
  static bool classof(const ASTNode *N) {
    return N->getKind() == NK_If;
  }

  ASTNode *getThen() {
    return Then;
  }

  ASTNode *getElse() {
    return Else;
  }

  void setThen(ASTNode *Node) {
    Then = Node;
  }

  void setElse(ASTNode *Node) {
    Else = Node;
  }

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

};

class ScsNode : public ASTNode {

private:
  ASTNode *Body;

public:
  ScsNode(BasicBlockNode *CFGNode, ASTNode *Body) :
    ASTNode(NK_Scs, CFGNode), Body(Body) {}
  ScsNode(BasicBlockNode *CFGNode, ASTNode *Body, ASTNode *Successor) :
    ASTNode(NK_Scs, CFGNode, Successor), Body(Body) {}

public:
  static bool classof(const ASTNode *N) {
    return N->getKind() == NK_Scs;
  }

  ASTNode *getBody() {
    return Body;
  }

  void setBody(ASTNode *Node) {
    Body = Node;
  }

  bool isEqual(ASTNode *Node);

  void dump(std::ofstream &ASTFile);

  BasicBlockNode *getFirstCFG();

};

class SequenceNode : public ASTNode {

public:
  using links_container  = std::vector<ASTNode *>;
  using links_iterator = typename links_container::iterator;
  using links_range = llvm::iterator_range<links_iterator>;

private:
  links_container NodeList;

public:
  SequenceNode(std::string Name) : ASTNode(NK_List, Name) {}
  SequenceNode(BasicBlockNode *CFGNode) : ASTNode(NK_List, CFGNode) {}

public:
  static bool classof(const ASTNode *N) {
    return N->getKind() == NK_List;
  }

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

  int listSize() {
    return NodeList.size();
  }

  ASTNode *getNodeN(int N) {
    return NodeList[N];
  }

  bool isEqual(ASTNode *Node);

  void dump(std::ofstream &ASTFile);

  BasicBlockNode *getFirstCFG();

};

#endif // define REVNGC_RESTRUCTURE_CFG_ASTNODE_H
