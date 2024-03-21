/// \file Visibility.cpp
/// \brief Calculate visibility information from PageBuster output.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <set>
#include <string>
#include <iostream>
#include <algorithm>

#include "revng/Model/Pass/RegisterModelPass.h"
#include "revng/Model/Pass/Visibility.h"
#include "revng/ADT/GenericGraph.h"

#include "llvm/Support/GraphWriter.h"

using namespace llvm;
using namespace model;

static RegisterModelPass R("visibility",
                           "Calculate visibility information from PageBuster "
                           "output",
                           model::calculateVisibility);

/* Time Span
 *
 * Struct to model the whole lifetime of a memory dump, ranging from its
 * starting epoch to the ending epoch. Also, a pointer to the model::Segment
 * it's saved.
 */
struct Span {

  // Reference to Model Segment
  const model::Segment *Segment;

  // Starting epoch
  unsigned long StartEpoch;

  // Ending epoch
  unsigned long EndEpoch;
};

/* Graph Node
 *
 * We build a graph where each node is a Span, and its outgoing edges represent
 * its forward visibility
 */
struct VisibilityNode {
  VisibilityNode(Span Node) {
    this->Node = Node;
  }
  Span Node;
};

using NodeType = ForwardNode<VisibilityNode>;
using VisibilityMap = GenericGraph<NodeType>;

static bool compareByEpoch(const Span &A, const Span &B)
{
    return A.StartEpoch < B.StartEpoch;
}

static bool allEpochsArePresent(vector<Span> Vec)
{
  // We want to scan from the second element onwards
  for (unsigned long I = 1; I < Vec.size(); I++) {
    if (Vec[I].StartEpoch > Vec[I-1].StartEpoch + 1)
      return false;
  }
  return true;
}

template<>
struct llvm::DOTGraphTraits<VisibilityMap *>
: public llvm::DefaultDOTGraphTraits {
  using EdgeIterator = llvm::GraphTraits<VisibilityMap *>::ChildIteratorType;
  DOTGraphTraits(bool IsSimple = false) : DefaultDOTGraphTraits(IsSimple) {}

  static std::string
  getNodeLabel(const NodeType *Node, const VisibilityMap *Graph) {
    std::stringstream Stream;
    Stream << "0x"
         << std::hex << Node->data().Node.Segment->StartAddress.address();
    return Stream.str();
  }

  static std::string getEdgeAttributes(const NodeType *Node,
                                       const EdgeIterator EI,
                                       const VisibilityMap *Graph) {
    return "color=black,style=dashed";
  }
};

void model::calculateVisibility(TupleTree<model::Binary> &Model) {

  /* Vector of Lifetimes
   *
   * At this stage, the model has segments with the lifetime information.
   * With a scan, we want to populate a vector of segments' lifespans.
   *
   * Each entry in the vector, has a reference to the model segment,
   * the starting epoch and the ending epoch for this particular memory
   * dump.
   *
   */
  vector<Span> SegmentsSpans;

  /*
   * Populate the vector as we scan the Model
   */
  for (const model::Segment &Segment : Model->Segments) {
    Span Entry;

    Entry.Segment = &Segment;
    Entry.StartEpoch = Segment.StartAddress.epoch();
    Entry.EndEpoch = Segment.Lifetime;

    SegmentsSpans.push_back(Entry);
  }

  /* Sort the Vector
   *
   * We have no explicit order of insertion guaranteed in the Model, so
   * that we are forced to reorder it, according to the StartEpoch.
   */
  std::sort(SegmentsSpans.begin(), SegmentsSpans.end(), compareByEpoch);

  /* Epochs check
   *
   * Assert that all Epoch's values are present
   */
  assert(allEpochsArePresent(SegmentsSpans));

  /* Map of Visibility
   *
   * Scan the vector and for each element (segment) compute all the
   * visible segments.
   *
   * At this stage, a segment is visible from another one if its lifespan
   * somehow overlaps with the other's lifes. In general two memory dumps
   * see each other if their lifespans overlaps. Visibility is intrinsecally
   * reciprocal.
   *
   */
  VisibilityMap VM;

  /* Active Set
   *
   * This set will contain pointers to Graph nodes that are
   * active during the scan of the vector.
   */
  using NodeSet = std::set<NodeType *>;
  NodeSet ActiveSet;

  /* Current Epoch
   *
   * This value is used to track which is the last epoch encountered
   * while scanning the vector. Sometimes some entries can be splitted
   * so we can have more than one segment (entry) with the same epoch
   */
  unsigned long CurrentEpoch = 1;

  for (const Span &Entry : SegmentsSpans) {

    if (Entry.StartEpoch != CurrentEpoch) {

      // Update Current Epoch
      CurrentEpoch++;

      // Remove from ActiveSet all the entries that are no more visible,
      // a.k.a. "dead"
      //
      // TODO: Compact with `erase_if`
      for (auto It = ActiveSet.begin(); It != ActiveSet.end(); ) {
          if ((*It)->Node.EndEpoch == CurrentEpoch) {
              It = ActiveSet.erase(It);
          }
          else {
              ++It;
          }
      }
    }

    // Add node to the Visibility Map
    auto *NewNode = VM.addNode(VisibilityNode{Entry});

    // Make this segment visible from the active ones
    // Make all the active segments visible from this segment
    for (auto Active: ActiveSet) {
      auto TmpNode = Active;
      TmpNode->addSuccessor(NewNode);
      NewNode->addSuccessor(TmpNode);
    }

    // Add this segment to active set
    ActiveSet.insert(NewNode);
  }

  llvm::raw_os_ostream Stream(dbg);
  llvm::WriteGraph(Stream, &VM, "debug");

  llvm::ViewGraph(&VM, "dot_graph");

  return;
}
