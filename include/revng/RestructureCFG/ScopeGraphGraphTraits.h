#pragma once

//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
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
/// edges as actual edges, and ignoring the goto edges
template<class GraphType>
struct Scope {

  // Special care should be employed when instantiating this class with a `*`
  // `GraphType` (like we do with `Function *`), because the `const GraphType &`
  // will be a temporary which can go out of scope as soon as the constructor
  // terminates. We do it this way in order to mimic what it is done in
  // `llvm/ADT/GraphTraits.h` for the `Inverse` marker class.
  const GraphType &Graph;

  inline Scope(const GraphType &G) : Graph(G) {}

  // Delete the constructor accepting a rvalue reference, in order to avoid
  // ill-formed instances of the class
  Scope(const GraphType &&) = delete;
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
  llvm::Function *PF = &F;
  llvm::dbgs() << "Depth first order:\n";
  for (auto *DFSNode : llvm::depth_first(Scope<llvm::Function *>(PF))) {
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
}
