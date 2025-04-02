#pragma once

//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/DOTGraphTraits.h"
#include "llvm/Support/GenericDomTree.h"
#include "llvm/Support/GraphWriter.h"

#include "revng/ADT/Concepts.h"
#include "revng/ADT/EagerMaterializationRangeIterator.h"
#include "revng/RestructureCFG/ScopeGraphUtils.h"
#include "revng/Support/FunctionTags.h"
#include "revng/Support/Tag.h"
#include "revng/Yield/FunctionEdgeType.h"

// We use a template here in order to instantiate `BlockType` both as
// `BasicBlock *` and `const BasicBlock *`
template<ConstOrNot<llvm::BasicBlock> BlockType>
inline llvm::SmallVector<BlockType *> getScopeGraphSuccessors(BlockType *BB) {

  llvm::SmallVector<BlockType *> Successors;
  //  First of all, we return all the standard successors of `BB`, but only if
  //  the current block does not contain the `goto_block` marker. If that is the
  //  case, since we have the constraint that a `goto_block` can only exists in
  //  a block with a single successor (which is the `goto` target).
  if (not isGotoBlock(BB))
    for (BlockType *S : successors(BB))
      Successors.push_back(S);

  // We then move to returning the additional successor represented by the
  // `ScopeCloser` edge, if present at all
  if (BlockType *ScopeCloserTarget = getScopeCloserTarget(BB))
    Successors.push_back(ScopeCloserTarget);

  return Successors;
}

template<ConstOrNot<llvm::BasicBlock> BlockType>
inline llvm::SmallVector<BlockType *> getScopeGraphPredecessors(BlockType *BB) {

  llvm::SmallVector<BlockType *> Predecessors;

  // We iterate over all the standard predecessors of `BB`. If one of the
  // predecessors has the `goto_block` marker, we skip such predecessor. This
  // can be done since we have the constraint that a `goto_block` can only
  // exists in a block with a single successor (which must be `BB` in our case).
  for (BlockType *S : predecessors(BB)) {
    if (not isGotoBlock(S)) {
      Predecessors.push_back(S);
    }
  }

  // We search for predecessors reaching `BB` through a `scope_closer` edge, by
  // iterating over the uses of the `BB`, filtering the uses which are
  // `BlockAddresses` that are used as arguments inside a `scope_closer` call
  for (auto *User : BB->users()) {
    if (auto *BA = llvm::dyn_cast<llvm::BlockAddress>(User)) {
      for (auto *BAUser : BA->users()) {
        if (auto *Call = getCallToTagged(BAUser,
                                         FunctionTags::ScopeCloserMarker)) {
          BlockType *CallParent = Call->getParent();
          Predecessors.push_back(CallParent);
        }
      }
    }
  }

  return Predecessors;
}

/// This class is used as a marker class to tell the graph iterator to treat the
/// underlying graph as a scope graph, i.e., considering also the scope closer
/// edges as actual edges, and ignoring the goto edges. We require `GraphType`
/// to be a pointer, so that we know we can non expensively copy it inside the
/// `Scope` struct, and avoid using the `const GraphType &` paradigm using,
/// e.g., in the `Inverse` marker class, which may cause "temporary out of
/// scope" bugs.
template<class GraphType>
  requires std::is_pointer_v<GraphType>
struct Scope {
  const GraphType Graph;

  inline Scope(const GraphType G) : Graph(G) {}
};

/// Specializes `GraphTraits<Scope<llvm::BasicBlock *>>`
template<>
struct llvm::GraphTraits<Scope<llvm::BasicBlock *>> {
public:
  using NodeRef = llvm::BasicBlock *;
  using ChildIteratorType = EagerMaterializationRangeIterator<
    llvm::BasicBlock *>;

public:
  static ChildIteratorType child_begin(NodeRef N) {
    return EagerMaterializationRangeIterator<
      llvm::BasicBlock *>(getScopeGraphSuccessors(N));
  }

  static ChildIteratorType child_end(NodeRef N) {
    return EagerMaterializationRangeIterator<llvm::BasicBlock *>::end();
  }

  // In the implementation for `llvm::BasicBlock *` trait we simply return
  // `this`
  static NodeRef getEntryNode(Scope<NodeRef> N) { return N.Graph; }

  // Add a verify method to the trait which checks that we have at maximum one
  // occurrence of the marker call in each `BasicBlock`, in the correct position
  static void verify(NodeRef N) { verifyScopeGraphAnnotations(N); }
};

template<>
struct llvm::GraphTraits<Scope<const llvm::BasicBlock *>> {
public:
  using NodeRef = const llvm::BasicBlock *;
  using ChildIteratorType = EagerMaterializationRangeIterator<
    const llvm::BasicBlock *>;

public:
  static ChildIteratorType child_begin(NodeRef N) {
    return EagerMaterializationRangeIterator<
      const llvm::BasicBlock *>(getScopeGraphSuccessors(N));
  }

  static ChildIteratorType child_end(NodeRef N) {
    return EagerMaterializationRangeIterator<const llvm::BasicBlock *>::end();
  }

  // In the implementation for `llvm::BasicBlock *` trait we simply return
  // `this`
  static NodeRef getEntryNode(Scope<NodeRef> N) { return N.Graph; }

  // Add a verify method to the trait which checks that we have at maximum one
  // occurrence of the marker call in each `BasicBlock`, in the correct position
  static void verify(NodeRef N) { verifyScopeGraphAnnotations(N); }
};

/// Specializes `GraphTraits<llvm::Inverse<Scope<llvm::BasicBlock *>>>`, this is
/// needed to compute the `PostDominatorTree`
template<>
struct llvm::GraphTraits<llvm::Inverse<Scope<llvm::BasicBlock *>>> {
public:
  using NodeRef = llvm::BasicBlock *;
  using ChildIteratorType = EagerMaterializationRangeIterator<
    llvm::BasicBlock *>;

public:
  static ChildIteratorType child_begin(NodeRef N) {
    return EagerMaterializationRangeIterator<
      llvm::BasicBlock *>(getScopeGraphPredecessors(N));
  }

  static ChildIteratorType child_end(NodeRef N) {
    return EagerMaterializationRangeIterator<llvm::BasicBlock *>::end();
  }

  // In the implementation for `llvm::BasicBlock *` trait we simply return
  // `this`
  static NodeRef getEntryNode(llvm::Inverse<Scope<NodeRef>> N) {
    return N.Graph.Graph;
  }

  // Add a verify method to the trait which checks that we have at maximum one
  // occurrence of the marker call in each `BasicBlock`, in the correct position
  static void verify(NodeRef N) { verifyScopeGraphAnnotations(N); }
};

template<>
struct llvm::GraphTraits<llvm::Inverse<Scope<const llvm::BasicBlock *>>> {
public:
  using NodeRef = const llvm::BasicBlock *;
  using ChildIteratorType = EagerMaterializationRangeIterator<
    const llvm::BasicBlock *>;

public:
  static ChildIteratorType child_begin(NodeRef N) {
    return EagerMaterializationRangeIterator<
      const llvm::BasicBlock *>(getScopeGraphPredecessors(N));
  }

  static ChildIteratorType child_end(NodeRef N) {
    return EagerMaterializationRangeIterator<const llvm::BasicBlock *>::end();
  }

  // In the implementation for `llvm::BasicBlock *` trait we simply return
  // `this`
  static NodeRef getEntryNode(llvm::Inverse<Scope<NodeRef>> N) {
    return N.Graph.Graph;
  }

  // Add a verify method to the trait which checks that we have at maximum one
  // occurrence of the marker call in each `BasicBlock`, in the correct position
  static void verify(NodeRef N) { verifyScopeGraphAnnotations(N); }
};

template<>
struct llvm::GraphTraits<Scope<llvm::Function *>>
  : public llvm::GraphTraits<Scope<typename llvm::BasicBlock *>> {
  using NodeRef = llvm::BasicBlock *;
  using nodes_iterator = pointer_iterator<Function::iterator>;

  static NodeRef getEntryNode(Scope<llvm::Function *> G) {
    return &G.Graph->getEntryBlock();
  }

  static nodes_iterator nodes_begin(Scope<llvm::Function *> G) {
    return nodes_iterator(G.Graph->begin());
  }

  static nodes_iterator nodes_end(Scope<llvm::Function *> G) {
    return nodes_iterator(G.Graph->end());
  }

  static size_t size(Scope<llvm::Function *> G) { return G.Graph->size(); }

  // Add a verify method to the trait which invokes the `verify` of the
  // `BasicBlock *` trait for each node in the graph
  static void verify(Scope<llvm::Function *> G) {
    for (auto &N : *G.Graph) {
      llvm::GraphTraits<Scope<llvm::BasicBlock *>>::verify(&N);
    }
  }
};

template<>
struct llvm::GraphTraits<Scope<const llvm::Function *>>
  : public llvm::GraphTraits<Scope<const typename llvm::BasicBlock *>> {
  using NodeRef = const llvm::BasicBlock *;
  using nodes_iterator = pointer_iterator<Function::const_iterator>;

  static NodeRef getEntryNode(Scope<const llvm::Function *> G) {
    return &G.Graph->getEntryBlock();
  }

  static nodes_iterator nodes_begin(Scope<const llvm::Function *> G) {
    return nodes_iterator(G.Graph->begin());
  }

  static nodes_iterator nodes_end(Scope<const llvm::Function *> G) {
    return nodes_iterator(G.Graph->end());
  }

  static size_t size(Scope<const llvm::Function *> G) {
    return G.Graph->size();
  }

  // Add a verify method to the trait which invokes the `verify` of the
  // `BasicBlock *` trait for each node in the graph
  static void verify(Scope<const llvm::Function *> G) {
    for (auto &N : *G.Graph) {
      llvm::GraphTraits<Scope<const llvm::BasicBlock *>>::verify(&N);
    }
  }
};

template<>
struct llvm::GraphTraits<llvm::Inverse<Scope<llvm::Function *>>>
  : public llvm::GraphTraits<
      llvm::Inverse<Scope<typename llvm::BasicBlock *>>> {
  using NodeRef = llvm::BasicBlock *;
  using nodes_iterator = pointer_iterator<Function::iterator>;

  static NodeRef getEntryNode(llvm::Inverse<Scope<llvm::Function *>> G) {
    return &G.Graph.Graph->getEntryBlock();
  }

  // Add a verify method to the trait which invokes the `verify` of the
  // `BasicBlock *` trait for each node in the graph
  static void verify(llvm::Inverse<Scope<llvm::Function *>> G) {
    for (auto &N : *G.Graph.Graph) {
      llvm::GraphTraits<llvm::Inverse<Scope<llvm::BasicBlock *>>>::verify(&N);
    }
  }
};

template<>
struct llvm::GraphTraits<llvm::Inverse<Scope<const llvm::Function *>>>
  : public llvm::GraphTraits<
      llvm::Inverse<Scope<const typename llvm::BasicBlock *>>> {
  using NodeRef = const llvm::BasicBlock *;
  using nodes_iterator = pointer_iterator<Function::const_iterator>;

  static NodeRef getEntryNode(llvm::Inverse<Scope<const llvm::Function *>> G) {
    return &G.Graph.Graph->getEntryBlock();
  }

  // Add a verify method to the trait which invokes the `verify` of the
  // `BasicBlock *` trait for each node in the graph
  static void verify(llvm::Inverse<Scope<const llvm::Function *>> G) {
    for (auto &N : *G.Graph.Graph) {
      llvm::GraphTraits<
        llvm::Inverse<Scope<const llvm::BasicBlock *>>>::verify(&N);
    }
  }
};

inline std::string getNodeLabel(const llvm::BasicBlock *N,
                                const llvm::Function *F) {
  if (N->hasName()) {
    return N->getName().str();
  } else {

    // We iterate over the function and we assign an incremental ID for each
    // unnamed `BasicBlock` that we encounter
    unsigned ID = 0;
    for (const auto *Node : llvm::nodes(F)) {
      if (N == Node) {
        return "block_" + std::to_string(ID);
      } else {
        ++ID;
      }
    }
  }

  revng_abort("We should always find the basicblock in the loop above");
}

inline std::string
getEdgeAttributes(const llvm::BasicBlock *N,
                  llvm::GraphTraits<
                    Scope<llvm::BasicBlock *>>::ChildIteratorType EI,
                  const Scope<llvm::Function *> G) {

  // If we are printing a block which has a `scope_closer` edge, which by
  // construction must be unique, if present, and the last in the
  // `MaterializedRange` representing the successors, we print such edge as
  // dashed
  using EMRI = EagerMaterializationRangeIterator<llvm::BasicBlock *>;
  if (getScopeCloserTarget(N) and std::next(EI) == EMRI::end()) {
    return "style=dashed";
  }

  // TODO: we may want to print `goto` edges as dotted lines, that do not affect
  //       the graph layout (`constraint=false`). This cannot be done inside
  //       this method, because the `goto` edges are not "iterated on" during
  //       the serialization of the `ScopeGraph`. We may want to specialize the
  //       `addCustomGraphFeatures` for such purposes.

  // If we excluded that the edge is scope_closer edge, we can print it
  // as a solid edge
  return "style=solid";
}

template<>
struct llvm::DOTGraphTraits<Scope<llvm::Function *>>
  : public llvm::DefaultDOTGraphTraits {
  using llvm::DefaultDOTGraphTraits::DefaultDOTGraphTraits;
  using EdgeIterator = llvm::GraphTraits<
    Scope<llvm::BasicBlock *>>::ChildIteratorType;

  std::string getNodeLabel(const llvm::BasicBlock *N,
                           const Scope<llvm::Function *> G) {
    return ::getNodeLabel(N, G.Graph);
  }

  std::string getEdgeAttributes(const llvm::BasicBlock *N,
                                const EdgeIterator EI,
                                const Scope<llvm::Function *> G) {
    return ::getEdgeAttributes(N, EI, G);
  }
};

template<>
struct llvm::DOTGraphTraits<Scope<const llvm::Function *>>
  : public llvm::DefaultDOTGraphTraits {
  using llvm::DefaultDOTGraphTraits::DefaultDOTGraphTraits;
  using EdgeIterator = llvm::GraphTraits<
    Scope<llvm::BasicBlock *>>::ChildIteratorType;

  std::string getNodeLabel(const llvm::BasicBlock *N,
                           const Scope<const llvm::Function *> G) {
    return ::getNodeLabel(N, G.Graph);
  }

  std::string getEdgeAttributes(const llvm::BasicBlock *N,
                                const EdgeIterator EI,
                                const Scope<llvm::Function *> G) {
    return ::getEdgeAttributes(N, EI, G);
  }
};

// Debug function used dump a serialized representation of the `ScopeGraph` on a
// stream
debug_function inline void dumpScopeGraph(llvm::Function &F) {
  llvm::dbgs() << "ScopeGraph of function: " << F.getName().str() << "\n";
  for (llvm::BasicBlock &BB : F) {
    llvm::dbgs() << "Block " << BB.getName().str() << " successors:\n";
    for (auto *Succ : llvm::children<Scope<llvm::BasicBlock *>>(&BB)) {
      llvm::dbgs() << " " << Succ->getName().str() << "\n";
    }
  }

  // Iteration on the whole graph using a `llvm::depth_first` visit
  llvm::dbgs() << "Depth first order:\n";
  for (auto *DFSNode : llvm::depth_first(Scope<llvm::Function *>(&F))) {
    llvm::dbgs() << DFSNode->getName().str() << "\n";
  }

  // Print the dominator tree
  llvm::DomTreeOnView<llvm::BasicBlock, Scope> DominatorTree;
  DominatorTree.recalculate(F);
  DominatorTree.print(llvm::dbgs());

  // Print the postdominator tree
  llvm::PostDomTreeOnView<llvm::BasicBlock, Scope> PostDominatorTree;
  PostDominatorTree.recalculate(F);
  PostDominatorTree.print(llvm::dbgs());

  // We also dump the `.dot` serialization of the `ScopeGraph` for debugging
  // purposes
  llvm::dbgs() << "Serializing the dot file representing the ScopeGraph\n";
  llvm::WriteGraph<Scope<llvm::Function *>>(&F, "ScopeGraph-" + F.getName());
}
