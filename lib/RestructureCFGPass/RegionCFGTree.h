#ifndef REGIONCFGTREE_H
#define REGIONCFGTREE_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <cstdlib>

// Local libraries includes
#include "revng/Support/Transform.h"

// Local includes
#include "ASTTree.h"

inline BasicBlockNode *getPointer(std::unique_ptr<BasicBlockNode> &Original) {
  return Original.get();
}

/// \brief The CFG, a container for BasicBlockNodes
class CFG {

public:
  using links_container = std::vector<std::unique_ptr<BasicBlockNode>>;
  using links_underlying_iterator = typename links_container::iterator;
  using links_iterator = TransformIterator<BasicBlockNode *,
                                           links_underlying_iterator>;
  using links_range = llvm::iterator_range<links_iterator>;

  links_iterator begin() { return links_iterator(BlockNodes.begin(),
                                                 getPointer); };
  links_iterator end() { return links_iterator(BlockNodes.end(),
                                                 getPointer); };

private:
  /// Storage for basic block nodes, associated to their original counterpart
  ///
  links_container BlockNodes;
  std::map<llvm::BasicBlock *, BasicBlockNode *> BBMap;

  /// Pointer to the entry basic block of this function
  llvm::BasicBlock *Entry;
  BasicBlockNode *EntryNode;
  std::map<BasicBlockNode *, BasicBlockNode *> SubstitutionMap;
  ASTTree AST;
  int IDCounter = 0;

public:

  CFG(std::set<BasicBlockNode *> &Nodes);

  CFG();

  void initialize(llvm::Function &F);

  std::string getID();

  links_range nodes() {
    return llvm::make_range(begin(), end());
  }

  size_t size();

  void setSize(int Size);

  void addNode(llvm::BasicBlock *BB);

  BasicBlockNode *newNode(std::string Name);

  BasicBlockNode *newNodeID(std::string Name);

  BasicBlockNode *newDummyNode(std::string Name);

  BasicBlockNode *newDummyNodeID(std::string Name);

  void removeNode(BasicBlockNode *Node);

  void insertBulkNodes(std::set<BasicBlockNode *> &Nodes,
                       BasicBlockNode *Head);

  void connectBreakNode(std::set<std::pair<BasicBlockNode *,
                                           BasicBlockNode *>> &Outgoing,
                        BasicBlockNode *Break);

  void connectContinueNode(BasicBlockNode *Continue);

  BasicBlockNode &get(llvm::BasicBlock *BB);

  BasicBlockNode &getEntryNode();

  std::vector<std::unique_ptr<BasicBlockNode>> &getNodes();

  BasicBlockNode &getRandomNode();

  std::vector<BasicBlockNode *> orderNodes(std::vector<BasicBlockNode *> &List,
                                           bool DoReverse);

  /// \brief Dump a GraphViz file on stdout representing this function
  void dumpDot();

  void purgeDummies();

  void purgeVirtualSink(BasicBlockNode *Sink);

  std::vector<BasicBlockNode *> getInterestingNodes(BasicBlockNode *Condition);

  void inflate();

  ASTNode *generateAst();

  // Get reference to the AST object which is inside the CFG object
  ASTTree &getAST();
};

// Provide graph traits for usage with, e.g., llvm::ReversePostOrderTraversal
namespace llvm {

template<> struct GraphTraits<CFG *> : public GraphTraits<BasicBlockNode *> {
  using nodes_iterator = CFG::links_iterator;

  static NodeRef getEntryNode(CFG *F) {
    return &F->getEntryNode();
  }

  static nodes_iterator nodes_begin(CFG *F) {
    return F->begin();
  }

  static nodes_iterator nodes_end(CFG *F) {
    return F->end();
  }

  static size_t size(CFG *F) { return F->size(); }
};

template<> struct GraphTraits<Inverse<CFG *>> :
  public GraphTraits<Inverse<BasicBlockNode *>> {

  static NodeRef getEntryNode(Inverse<CFG *> G) {
    return &G.Graph->getEntryNode();
  }
};

} // namespace llvm

#endif // REGIONCFGTREE_H
