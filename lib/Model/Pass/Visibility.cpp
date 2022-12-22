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

using namespace llvm;
using namespace model;

static RegisterModelPass R("visibility",
                           "Calculate visibility information from PageBuster "
                           "output",
                           model::calculateVisibility);

struct Span {

  // Reference to Model Segment
  const model::Segment *Segment;

  // Starting epoch
  unsigned long StartEpoch;

  // Ending epoch
  unsigned long EndEpoch;
};

bool compareByEpoch(const Span &A, const Span &B)
{
    return A.StartEpoch < B.StartEpoch;
}

bool allEpochsArePresent(vector<Span> Vec)
{
  // We want to scan from the second element onwards
  for (int i = 1; i < Vec.size(); i++) {
    if (Vec[i].StartEpoch > Vec[i-1].StartEpoch + 1)
      return false;
  }
  return true;
}

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
    Span* Entry;

    Entry.Segment = &Segment;
    Entry.StartEpoch = Segment.StartAddress.epoch();
    Entry.EndEpoch = Segment.Lifetime;

    SegmentsSpans.push_back(*Entry);
  }

  /* Sort the Vector
   *
   * We have no explicit order of insertion guaranteed in the Model, so
   * that we are forced to reorder it, according to the StartEpoch.
   */
  std::sort(SegmentsSpans.begin(), SegmentsSpans.end(), compareByEpoch);

  /* Epoch check
   *
   * Assert that all Epoch's values are present
   */
  assert(allEpochsArePresent(SegmentsSpans));

  for (const Span &Entry : SegmentsSpans)
    std::cout << Entry.StartEpoch << ' ';

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
  struct VisibilityNode {
    VisibilityNode(Span) {}
    Span Node;

  };
  using NodeType = ForwardNode<VisibilityNode>;
  using VisibilityMap = GenericGraph<NodeType>;
  VisibilityMap VM;

  /*
   * Create all the Visibility Nodes
   */
  // TODO

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

      // Remove from ActiveSet all the entries that are no more visible,
      // a.k.a. "dead"
      for (auto It = ActiveSet.begin(); It != ActiveSet.end(); ) {
          if ((*It)->Node.EndEpoch == CurrentEpoch) {
              It = ActiveSet.erase(It);
          }
          else {
              ++It;
          }
      }

      // Update Current Epoch
      CurrentEpoch++;
    }

    // Add node to the Visibility Map
    auto *NewNode = VM.addNode(Entry);

    // Make this segment visible from the active ones
    for (auto Active: ActiveSet) {
      // auto TmpNode = Active.
      TmpNode.addSuccessor(NewNode);
    }

    // Make all the active segments visible from this segment
    for (auto Active: ActiveSet) {
      NewNode.addSuccessor(TmpNode);
    }

    // Add this segment to active set
    ActiveSet.insert(NewNode);
  }

  return;
}
