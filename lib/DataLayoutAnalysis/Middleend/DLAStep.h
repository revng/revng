#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <algorithm>
#include <memory>
#include <type_traits>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#include "revng-c/DataLayoutAnalysis/DLATypeSystem.h"

namespace dla {

// Forward declaration for dla::LayoutTypeSystem, that will be defined later.
// For this preliminary implementation, dla::Step does not really need the
// implementation of the dla::LayoutTypeSystem class, because for now we are
// only concerned about the schedule of the Steps, and the Step implementations
// are really just empty.
class LayoutTypeSystem;

class Step {
public:
  using IDSet = llvm::SmallPtrSet<const void *, 2>;
  using IDSetRef = llvm::SmallPtrSetImpl<const void *> &;
  using IDSetConstRef = const llvm::SmallPtrSetImpl<const void *> &;

protected:
  const void *StepID;

  // TODO: rework and check dependencies and invalidations
  IDSet Dependencies;
  IDSet Invalidated;

  Step(const char &C,
       std::initializer_list<const void *> D,
       std::initializer_list<const void *> I) :
    StepID(&C), Dependencies(D), Invalidated(I) {}

  Step(const char &C) : Step(C, {}, {}) {}

public:
  Step() = delete;
  virtual ~Step() = default;

  /// Runs the Step on TS, returns true if it has applied changes to TS.
  virtual bool runOnTypeSystem(LayoutTypeSystem &TS) = 0;

  IDSetConstRef getDependencies() const { return Dependencies; }
  IDSetConstRef getInvalidated() const { return Invalidated; }

  const void *getStepID() const { return StepID; };
};

/// Collapses strongly connected components made of equality edges
//
// After the execution of this step, the LayoutTypeSystem graph should not
// contain equality edges anymore
class CollapseEqualitySCC : public Step {
  static const char ID;

public:
  static const constexpr void *getID() { return &ID; }

  CollapseEqualitySCC() : Step(ID){};

  virtual ~CollapseEqualitySCC() override = default;

  virtual bool runOnTypeSystem(LayoutTypeSystem &TS) override;
};

/// Collapses strongly connected components in the type system made of
/// instance-at-offset-0 edges
class CollapseInstanceAtOffset0SCC : public Step {
  static const char ID;

public:
  static const constexpr void *getID() { return &ID; }

  CollapseInstanceAtOffset0SCC() : Step(ID){};

  virtual ~CollapseInstanceAtOffset0SCC() override = default;

  virtual bool runOnTypeSystem(LayoutTypeSystem &TS) override;
};

/// dla::Step that simplifies instance-at-offset-0 edges, to reduce the
/// unnecessary layers of nested types
class SimplifyInstanceAtOffset0 : public Step {
  static const char ID;

public:
  static const constexpr void *getID() { return &ID; }

  SimplifyInstanceAtOffset0() :
    Step(ID,
         // Dependencies
         { CollapseInstanceAtOffset0SCC::getID() },
         // Invalidated
         {}) {}

  virtual ~SimplifyInstanceAtOffset0() override = default;

  virtual bool runOnTypeSystem(LayoutTypeSystem &TS) override;
};

/// dla::Step that removes leaf nodes without valid layout information
//
// Initially valid layout information is simply represented by accesses, but we
// expect this to be possibly user provided for leafs that otherwise had no
// valid layout information, such as calls to external library functions.
class PruneLayoutNodesWithoutLayout : public Step {
  static const char ID;

public:
  static const constexpr void *getID() { return &ID; }

  PruneLayoutNodesWithoutLayout() : Step(ID) {}

  virtual ~PruneLayoutNodesWithoutLayout() override = default;

  virtual bool runOnTypeSystem(LayoutTypeSystem &TS) override;
};

/// dla::Step that merge pointer nodes pointing to the same layout
class MergePointerNodes : public Step {
  static const char ID;

public:
  static const constexpr void *getID() { return &ID; }

  MergePointerNodes() : Step(ID) {}

  virtual ~MergePointerNodes() override = default;

  virtual bool runOnTypeSystem(LayoutTypeSystem &TS) override;
};

/// dla::Step that takes all strided edges and decompose in edges with only one
/// stride layer
class DecomposeStridedEdges : public Step {
  static const char ID;

public:
  static const constexpr void *getID() { return &ID; }

  inline DecomposeStridedEdges();

  virtual ~DecomposeStridedEdges() override = default;

  virtual bool runOnTypeSystem(LayoutTypeSystem &TS) override;
};

/// dla::Step that computes and propagates information on accesses and type
/// sizes.
class ComputeUpperMemberAccesses : public Step {

  static const char ID;

public:
  static const constexpr void *getID() { return &ID; }

  ComputeUpperMemberAccesses() :
    Step(ID,
         // Dependencies
         {
           CollapseInstanceAtOffset0SCC::getID(),
           CollapseEqualitySCC::getID(),
           PruneLayoutNodesWithoutLayout::getID(),
         },
         // Invalidated
         {}) {}

  virtual ~ComputeUpperMemberAccesses() override = default;

  virtual bool runOnTypeSystem(LayoutTypeSystem &TS) override;
};

/// dla::Step that removes invalid stride edges
class RemoveInvalidStrideEdges : public Step {
  static const char ID;

public:
  static const constexpr void *getID() { return &ID; }

  RemoveInvalidStrideEdges() :
    Step(ID,
         // Dependencies
         { ComputeUpperMemberAccesses::getID() },
         // Invalidated
         { ComputeUpperMemberAccesses::getID(),
           PruneLayoutNodesWithoutLayout::getID() }) {}

  virtual ~RemoveInvalidStrideEdges() override = default;

  virtual bool runOnTypeSystem(LayoutTypeSystem &TS) override;
};

/// dla::Step that merge pointee nodes of union of pointers
class MergePointeesOfPointerUnion : public Step {
  static const char ID;

  size_t PointerSize;

public:
  static const constexpr void *getID() { return &ID; }

  MergePointeesOfPointerUnion(size_t PtrSize) :
    Step(ID,
         // Dependencies
         { ComputeUpperMemberAccesses::getID() },
         // Invalidated
         { ComputeUpperMemberAccesses::getID(),
           CollapseInstanceAtOffset0SCC::getID(),
           RemoveInvalidStrideEdges::getID() }),
    PointerSize(PtrSize) {}

  virtual ~MergePointeesOfPointerUnion() override = default;

  virtual bool runOnTypeSystem(LayoutTypeSystem &TS) override;
};

/// dla::Step that collapses nodes that have a single child at offset 0
class CollapseSingleChild : public Step {
  static const char ID;

public:
  static const constexpr void *getID() { return &ID; }
  static bool collapseSingle(LayoutTypeSystem &TS, LayoutTypeSystemNode *Node);

  CollapseSingleChild() :
    Step(ID,
         // Dependencies
         { ComputeUpperMemberAccesses::getID() },
         // Invalidated
         {}) {}

  virtual ~CollapseSingleChild() override = default;

  virtual bool runOnTypeSystem(LayoutTypeSystem &TS) override;
};

/// dla::Step that decompose the LayoutTypeSystem into components, each of which
/// cannot overlap with others
class ComputeNonInterferingComponents : public Step {
  static const char ID;

public:
  static const constexpr void *getID() { return &ID; }

  ComputeNonInterferingComponents() :
    Step(ID,
         // Dependencies
         { ComputeUpperMemberAccesses::getID() },
         // Invalidated
         {}) {}

  virtual ~ComputeNonInterferingComponents() override = default;

  virtual bool runOnTypeSystem(LayoutTypeSystem &TS) override;
};

/// dla::Step that removes invalid pointer edges
class RemoveInvalidPointers : public Step {
  static const char ID;

  size_t PointerSize;

public:
  static const constexpr void *getID() { return &ID; }

  RemoveInvalidPointers(size_t PtrSize) : Step(ID), PointerSize(PtrSize) {}

  virtual ~RemoveInvalidPointers() override = default;

  virtual bool runOnTypeSystem(LayoutTypeSystem &TS) override;
};

/// dla::Step that tries to compact partly overlapping compatible arrays
class CompactCompatibleArrays : public Step {
  static const char ID;

public:
  static const constexpr void *getID() { return &ID; }

  CompactCompatibleArrays() :
    Step(ID,
         { ComputeUpperMemberAccesses::getID(),
           DecomposeStridedEdges::getID(),
           PruneLayoutNodesWithoutLayout::getID() },
         // Invalidated
         {}) {}

  virtual ~CompactCompatibleArrays() override = default;

  virtual bool runOnTypeSystem(LayoutTypeSystem &TS) override;
};

/// dla::Step that tries to pushes down instance edges that are actually part of
/// a child node.
class ArrangeAccessesHierarchically : public Step {
  static const char ID;

public:
  static const constexpr void *getID() { return &ID; }

  ArrangeAccessesHierarchically() :
    Step(ID,
         { ComputeUpperMemberAccesses::getID(),
           DecomposeStridedEdges::getID(),
           PruneLayoutNodesWithoutLayout::getID() },
         // Invalidated
         {}) {}

  virtual ~ArrangeAccessesHierarchically() override = default;

  virtual bool runOnTypeSystem(LayoutTypeSystem &TS) override;
};

/// dla::Step that tries to move pointer edges to push further down in the type
/// hierarchy.
class PushDownPointers : public Step {
  static const char ID;

public:
  static const constexpr void *getID() { return &ID; }

  PushDownPointers() : Step(ID) {}

  virtual ~PushDownPointers() override = default;

  virtual bool runOnTypeSystem(LayoutTypeSystem &TS) override;
};

/// dla::Step that resolves unions of primitive and pointer types
class ResolveLeafUnions : public Step {
  static const char ID;

public:
  static const constexpr void *getID() { return &ID; }

  ResolveLeafUnions() : Step(ID) {}

  virtual ~ResolveLeafUnions() override = default;

  virtual bool runOnTypeSystem(LayoutTypeSystem &TS) override;
};

/// dla::Step that merges structurally identical subtrees of an interfering
/// node.
class DeduplicateFields : public Step {
  static const char ID;

public:
  static const constexpr void *getID() { return &ID; }

  DeduplicateFields() : Step(ID) {}

  virtual ~DeduplicateFields() override = default;

  virtual bool runOnTypeSystem(LayoutTypeSystem &TS) override;
};

inline DecomposeStridedEdges::DecomposeStridedEdges() :
  Step(ID,
       // Dependencies
       { ComputeUpperMemberAccesses::getID() },
       // Invalidated
       { DeduplicateFields::getID() }) {
}

template<typename IterT>
bool intersect(IterT I1, IterT E1, IterT I2, IterT E2) {
  while ((I1 != E1) and (I2 != E2)) {
    if (*I1 < *I2)
      ++I1;
    else if (*I2 < *I1)
      ++I2;
    else
      return true;
  }
  return false;
}

template<typename RangeT>
bool intersect(const RangeT &R1, const RangeT &R2) {
  return intersect(R1.begin(), R1.end(), R2.begin(), R2.end());
}

class StepManager {

public:
  enum StepState {
    NewlyAdded,
    Invalidated,
    Done,
  };

public:
  llvm::SmallVector<std::unique_ptr<Step>, 16> Schedule;
  llvm::SmallPtrSet<const void *, 16> InsertedSteps;
  llvm::SmallPtrSet<const void *, 16> InvalidatedSteps;

  using sched_const_iterator = decltype(Schedule)::const_iterator;
  using sched_const_range = llvm::iterator_range<sched_const_iterator>;

public:
  StepManager() : Schedule(), InsertedSteps(), InvalidatedSteps() {}

  /// Adds a Step to the StepManager, moving ownership into it.
  [[nodiscard]] bool addStep(std::unique_ptr<Step> S);

  template<typename StepT, typename... ArgsT>
  [[nodiscard]] bool addStep(ArgsT &&...Args) {
    return addStep(std::make_unique<StepT>(std::forward<ArgsT &&>(Args)...));
  }

  /// Runs the added steps
  void run(LayoutTypeSystem &TS);

  /// Drops all the scheduled steps
  void reset() {
    Schedule.clear();
    InsertedSteps.clear();
    InvalidatedSteps.clear();
  }

  bool hasValidSchedule() const {
    return not intersect(InsertedSteps, InvalidatedSteps);
  }

  /// Get the number of scheduled steps
  auto getNumSteps() const { return Schedule.size(); }

  // Methods for const iteration on Schedule
  sched_const_iterator sched_begin() const { return Schedule.begin(); }
  sched_const_iterator sched_end() const { return Schedule.end(); }
  sched_const_range sched() const {
    return llvm::make_range(sched_begin(), sched_end());
  }
};

} // end namespace dla
