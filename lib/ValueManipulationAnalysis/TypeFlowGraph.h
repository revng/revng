#pragma once

//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include <set>

#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/ADT/FilteredGraphTraits.h"
#include "revng/ADT/GenericGraph.h"
#include "revng/EarlyFunctionAnalysis/FunctionMetadataCache.h"

#include "revng-c/ValueManipulationAnalysis/TypeColors.h"

#include "TypeFlowNode.h"

namespace vma {

// --------------- TypeFlowGraph

/// Graph representing how type information flows between values and uses
///
/// Nodes represent `llvm::Value`s and `llvm::Use`s and their candidate types,
/// edges represent how type information is propagated.
struct TypeFlowGraph : public GenericGraph<TypeFlowNode> {
  using GenericGraph::GenericGraph;

  TypeFlowNode *addNodeContaining(FunctionMetadataCache &Cache,
                                  const UseOrValue &);
  TypeFlowNode *getNodeContaining(const UseOrValue &) const;

  /// Print the graph on a `.dot` file
  /// \param Title title of the graph
  /// \param FileName if not specified, the default is `/tmp/<random string>`
  void dump(const llvm::Twine &Title = "",
            std::string FileName = "") debug_function;

  /// Dump a dot representation of the graph to the given stream
  void print(llvm::raw_ostream &OS) debug_function;

  /// Show the graph in a window, to be used inside a debugger
  void view() debug_function;

  const llvm::Function *Func;
  const model::Binary *Model;
  std::map<UseOrValue, TypeFlowNode *> ContentToNodeMap;
};

// --------------- TypeFlowGraph manipulation

/// Add to \a TG the `llvm::Use`s and `llvm::Value`s inside \a F
TypeFlowGraph makeTypeFlowGraphFromFunction(FunctionMetadataCache &Cache,
                                            const llvm::Function *F,
                                            const model::Binary *Model);

/// Propagate colors from colored nodes through colored edges
void propagateColors(TypeFlowGraph &TG);

/// Propagate a single color
template<unsigned Filter>
void propagateColor(TypeFlowGraph &TG);

/// Propagate Numberness and Pointerness with special rules
/// \return true if the TypeFlowGraph was modified
bool propagateNumberness(TypeFlowGraph &TG);

/// Make the graph undirected by adding all reciprocal edges
void makeBidirectional(TypeFlowGraph &TG);

/// Count the number of edges connecting nodes with disjoint candidates
unsigned countCasts(const TypeFlowGraph &TG);

/// Recursively assign grey nodes based on the color of their neighbors
///
/// See also TypeFlowNode::majorityVote()
/// \return True if at least one node was modified
bool applyMajorityVoting(TypeFlowGraph &TG);

// --------------- Filtered Graphs implementation

/// Select only edge that contain a given color
template<unsigned FilterColor>
inline bool hasColor(llvm::GraphTraits<TypeFlowGraph *>::EdgeRef &Edge) {
  return Edge.Colors.contains(FilterColor);
}

/// Filtered Graph with only edges of a given color
template<unsigned FilterColor>
using EdgeFilteredTG = EdgeFilteredGraph<TypeFlowNode *, hasColor<FilterColor>>;

/// Select only undecided nodes
inline bool
bothHaveManyCandidates(const llvm::GraphTraits<TypeFlowGraph *>::NodeRef &Src,
                       const llvm::GraphTraits<TypeFlowGraph *>::NodeRef &Tgt) {

  return Src->isUndecided() and Tgt->isUndecided();
}

/// Filtered Graph with only undecided nodes
using NodeFilteredTG = NodePairFilteredGraph<TypeFlowNode *,
                                             bothHaveManyCandidates>;

} // namespace vma
